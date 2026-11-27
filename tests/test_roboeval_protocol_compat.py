"""Wire compatibility with the current deployment, without importing two pb2 modules."""
import ast
from pathlib import Path

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

ROOT = Path(__file__).resolve().parents[1]


def descriptor(path):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'DESCRIPTOR' for t in node.targets))
    return descriptor_pb2.FileDescriptorProto.FromString(ast.literal_eval(assignment.value.args[0]))


def plan_class(doc):
    pool = descriptor_pool.DescriptorPool()
    pool.AddSerializedFile(doc.SerializeToString())
    return message_factory.GetMessageClass(pool.FindMessageTypeByName('colosseum.v1.ActionPlan'))


def descriptors():
    return descriptor(ROOT / 'tests/fixtures/protocol/upstream_colosseum_pb2.py'), descriptor(ROOT / 'src/colosseum_client/colosseum_pb2.py')


def test_every_existing_message_field_is_preserved():
    old, new = descriptors()
    messages = {m.name: m for m in new.message_type}
    for message in old.message_type:
        fields = {f.number: f for f in messages[message.name].field}
        for field in message.field:
            assert fields[field.number] == field
    assert next(m for m in new.message_type if m.name == 'ActionPlan').field[6].name == 'done'


def test_current_decoder_preserves_done_and_execution_actions():
    old, new = map(plan_class, descriptors())
    outgoing = new(plan_id=21, done=True, raw_action_source='test-only', raw_action_space='joint_position')
    outgoing.actions.shape.extend([1, 14])
    outgoing.actions.data = b'\0' * 56
    outgoing.raw_actions.shape.extend([30, 14])
    outgoing.raw_actions.data = b'\1' * 1680
    received = old.FromString(outgoing.SerializeToString())
    assert received.done is True
    assert received.actions.data == outgoing.actions.data
    assert list(received.actions.shape) == [1, 14]
    # Unknown evidence fields survive a legacy protobuf read/write cycle.
    restored = new.FromString(received.SerializeToString())
    assert restored.raw_actions == outgoing.raw_actions


def test_legacy_plan_has_missing_raw_not_fabricated_raw():
    old, new = map(plan_class, descriptors())
    received = new.FromString(old(plan_id=2, done=True).SerializeToString())
    assert received.done
    assert not received.HasField('raw_actions')
