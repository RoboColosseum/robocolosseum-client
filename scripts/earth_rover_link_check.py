"""Phase 0 link check: exercise the FrodoBots ride, Agora video, RTM telemetry and REST commands without the Router.

Run it on the operator machine with the rover online and an operator watching it:

    FRODOBOTS_SDK_TOKEN=... uv run --extra earth-rover python scripts/earth_rover_link_check.py \
        --bot-slug BOT [--mission-slug MISSION] [--drive] [--telemetry rtm|shadow] [--seconds 20]

Without --mission-slug it uses the SDK's token-only session (no ride, no platform recording).

Without --drive it only observes (frames, telemetry) and sends a confirmed stop.
With --drive it sends a slow forward command for one second, then a slow left turn,
so the angular sign and the command scale can be calibrated by eye.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

from colosseum_client.robots.earth_rover.commands import RtmRestSender
from colosseum_client.robots.earth_rover.session import DEFAULT_API_URL, FrodobotsSession


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--bot-slug', required=True)
    parser.add_argument('--mission-slug', default='', help='omit for a token-only session (no ride, no platform recording)')
    parser.add_argument('--api-url', default=DEFAULT_API_URL)
    parser.add_argument('--telemetry', choices=['rtm', 'shadow'], default='rtm')
    parser.add_argument('--seconds', type=float, default=20, help='observation window')
    parser.add_argument('--drive', action='store_true', help='send a short forward then left command (operator present)')
    parser.add_argument('--no-video', action='store_true', help='skip the Agora RTC subscription')
    parser.add_argument('--report', default='earth-rover-link-check.json')
    args = parser.parse_args()
    token = os.environ.get('FRODOBOTS_SDK_TOKEN')
    if not token:
        sys.exit('Set FRODOBOTS_SDK_TOKEN to the organizer sdk_access_token')
    report = {'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'telemetry_mode': args.telemetry}
    session = FrodobotsSession(token, args.bot_slug, args.mission_slug, api_url=args.api_url)
    video = telemetry = sender = None
    try:
        credentials = session.start_ride()
        report['ride'] = {'channel': credentials.channel, 'user_id': credentials.user_id,
                          'bot_uid': credentials.bot_uid, 'bot_type': credentials.bot_type}
        print(f"Ride acquired: channel={credentials.channel} user={credentials.user_id} bot={credentials.bot_uid}", flush=True)
        sender = RtmRestSender(credentials)
        if not args.no_video:
            from colosseum_client.robots.earth_rover.video import AgoraVideo
            video = AgoraVideo(credentials, ['front_image', 'rear_image'])
            video.start()
        if args.telemetry == 'rtm':
            from colosseum_client.robots.earth_rover.telemetry import RtmTelemetry
            telemetry = RtmTelemetry(credentials)
        else:
            from colosseum_client.robots.earth_rover.telemetry import ShadowTelemetry
            telemetry = ShadowTelemetry(session)
        telemetry.start()

        # Observe: frame ages and telemetry rate.
        frame_ages = {'front_image': [], 'rear_image': []}
        telemetry_stamps = []
        last_stamp = None
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            now = time.monotonic()
            if video is not None:
                for camera, (_, received) in video.latest().items():
                    frame_ages[camera].append(now - received)
            latest = telemetry.latest()
            if latest is not None and latest[1] != last_stamp:
                last_stamp = latest[1]
                telemetry_stamps.append(last_stamp)
            time.sleep(0.2)
        report['video'] = {camera: {'frames': video.store.counts.get(camera, 0) if video else 0,
                                    'median_age_s': statistics.median(ages) if ages else None,
                                    'max_age_s': max(ages) if ages else None}
                           for camera, ages in frame_ages.items()}
        gaps = [b - a for a, b in zip(telemetry_stamps, telemetry_stamps[1:])]
        latest = telemetry.latest()
        report['telemetry'] = {'reports': len(telemetry_stamps), 'median_interval_s': statistics.median(gaps) if gaps else None,
                               'last_values': None if latest is None else latest[0],
                               'login_code': getattr(telemetry, 'login_code', None), 'errors': getattr(telemetry, 'errors', [])[:5]}
        print(json.dumps({'video': report['video'], 'telemetry': report['telemetry']}, indent=2, default=str), flush=True)

        # Command round trip at 10 Hz (zeros unless --drive).
        latencies = []
        commands = []
        if args.drive:
            commands += [(0.2, 0.0)] * 10 + [(0.0, 0.0)] * 5 + [(0.0, 0.3)] * 10
            print('Driving: forward for 1 s, pause, then a left turn for 1 s. Watch the rover.', flush=True)
        commands += [(0.0, 0.0)] * 10
        for linear, angular in commands:
            started = time.monotonic()
            try:
                sender.send(linear, angular)
                latencies.append(time.monotonic() - started)
            except Exception as exc:
                latencies.append(None)
                print(f'send failed: {type(exc).__name__}', flush=True)
            time.sleep(max(0.0, 0.1 - (time.monotonic() - started)))
        measured = [value for value in latencies if value is not None]
        started = time.monotonic()
        confirmed = sender.stop(confirm=True)
        report['commands'] = {'sent': len(commands), 'failed': latencies.count(None),
                              'median_rtt_s': statistics.median(measured) if measured else None,
                              'max_rtt_s': max(measured) if measured else None,
                              'stop_confirmed': confirmed, 'stop_receipt_s': time.monotonic() - started,
                              'stop_error': sender.last_error}
        print(json.dumps(report['commands'], indent=2), flush=True)
    finally:
        for component in (telemetry, video):
            if component is not None:
                try:
                    component.close()
                except Exception as exc:
                    print(f'cleanup failed: {type(exc).__name__}', flush=True)
        if sender is not None:
            if session.credentials is not None and sender.stop(confirm=True):
                session.end_ride()
                report['ride_ended'] = True
            else:
                report['ride_ended'] = False
                print('Stop not confirmed: the ride was left open. Check the rover and end the ride from the app.', flush=True)
            sender.close()
        session.close()
        with open(args.report, 'w', encoding='utf-8') as output:
            json.dump(report, output, indent=2, default=str)
        print(f'Report written to {args.report}', flush=True)


if __name__ == '__main__':
    main()
