"""Independent CSV/pose audit of a completed measured-start control.py run.

Usage: python audit_settled_run.py part_b_logs/settled_transport_run_....json
Uses the actual goal saved by that run, never a hard-coded target or offset.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


def angle(a,b):
    # Quaternion conjugate(a)*b, xyzw convention; atan2 also resolves small angles.
    av,bv = a[:3],b[:3]
    cross = [av[1]*bv[2]-av[2]*bv[1],av[2]*bv[0]-av[0]*bv[2],
             av[0]*bv[1]-av[1]*bv[0]]
    vector = [a[3]*bv[i]-b[3]*av[i]-cross[i] for i in range(3)]
    scalar = a[3]*b[3]+sum(x*y for x,y in zip(av,bv))
    return 2.*math.atan2(math.sqrt(sum(x*x for x in vector)),abs(scalar))


def audit(filename):
    filename = Path(filename).resolve()
    report = json.loads(filename.read_text())
    root = filename.parent.parent
    assert report['completed'] and report['target_placement_evidence']
    assert report['fresh_planning'] and not report['cached_plan_used']
    assert not report['reference_offset_applied']
    for name,digest in report['source_sha256'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest, name+' changed since run'
    settling = report['settling']
    init = report['grasp_initialization']
    assert settling['settled'] and settling['robot_cube_contact_steps']==0
    assert not settling['cube_reset'] and not init['cube_reset']
    assert settling['stable_time_s']>=settling['required_stable_time_s']
    assert settling['elapsed_s']<=settling['timeout_s']+1e-12
    assert settling['linear_speed_m_s']<=settling['linear_speed_threshold_m_s']
    assert settling['angular_speed_rad_s']<=settling['angular_speed_threshold_rad_s']
    assert init['actual_cube_start_error_m']<1e-12
    assert init['actual_cube_start_orientation_error_rad']<1e-12
    assert init['actual_hand_position_error_m']<=1.1e-3
    assert init['actual_hand_orientation_error_rad']<=1.1e-3
    original = report['nominal_start_xyzquat']
    measured = report['actual_start_xyzquat']
    assert max(abs(measured[i]-original[i]-settling['settling_translation_m'][i])
               for i in range(3))<1e-12
    assert abs(angle(original[3:],measured[3:])-settling['settling_orientation_rad'])<1e-10
    assert report['preflight']['geometry_checked_before_timing']
    assert report['preflight']['required_robot_obstacle_clearance_m']==.002
    assert report['preflight']['required_cube_obstacle_clearance_m']==.008

    model = root/'models/nextagea_description/urdf/NextageaOpen.urdf'
    limits = {j.attrib['name']:{k:float(v) for k,v in j.find('limit').attrib.items()}
              for j in ET.parse(model).getroot().findall('joint') if j.find('limit') is not None}
    goal = report['original_goal_xyzquat']
    n = applied = clipped = final_count = 0
    previous_t = previous_tau = None
    peaks = dict.fromkeys(('obstacle_load_N','robot_table_load_N','cube_tracking_error_m',
                          'cube_tracking_orientation_error_rad','relative_slip_m'),0.)
    first = last = None
    active_grip_peak_slip = peak_joint_error = 0.
    phases = {}
    with Path(report['csv']).open(newline='') as stream:
        reader = csv.DictReader(stream)
        joints = [key[2:] for key in reader.fieldnames
                  if key.startswith('q_') and not key.startswith('q_des_')]
        assert len(joints)==15
        for row in reader:
            n += 1
            value = {key:float(x) for key,x in row.items() if key!='phase'}
            assert all(math.isfinite(x) for x in value.values()), (n,'nonfinite')
            t = value['time_s']
            assert abs(t if previous_t is None else t-previous_t-.001)<1e-10
            previous_t = t
            applied += int(value['step_allowed'])
            clipped += int(value['torque_clipped'])
            q_error = max(abs(value['q_'+j]-value['q_des_'+j]) for j in joints)
            peak_joint_error = max(peak_joint_error,q_error)
            phases[row['phase']] = phases.get(row['phase'],0)+1
            speed = max(abs(value['v_'+j]) for j in joints)
            assert abs(q_error-value['max_joint_error_rad'])<1e-12
            assert abs(speed-value['max_speed_rad_s'])<1e-12
            actual_tau = [value['tau_applied_previous_'+j] for j in joints]
            assert actual_tau==(previous_tau if previous_tau is not None else [0.]*len(joints))
            previous_tau = [value['tau_next_'+j] for j in joints]
            for j in joints:
                limit = limits[j]
                assert limit['lower']<=value['q_'+j]<=limit['upper']
                assert abs(value['v_'+j])<=limit['velocity']
                assert abs(value['tau_next_'+j])<=limit['effort']
                assert abs(value['tau_applied_previous_'+j])<=limit['effort']
            pos = [value['cube_'+a+'_m'] for a in 'xyz']
            desired = [value['cube_des_'+a+'_m'] for a in 'xyz']
            quat = [value['cube_q'+a] for a in 'xyzw']
            desired_quat = [value['cube_des_q'+a] for a in 'xyzw']
            assert abs(sum(x*x for x in quat)-1.)<1e-10
            assert abs(sum(x*x for x in desired_quat)-1.)<1e-10
            tracking = math.sqrt(sum((a-b)**2 for a,b in zip(pos,desired)))
            orientation = angle(desired_quat,quat)
            raw = math.sqrt(sum((a-b)**2 for a,b in zip(pos,goal[:3])))
            horizontal = math.hypot(pos[0]-goal[0],pos[1]-goal[1])
            raw_angle = angle(goal[3:],quat)
            # Every phase, including preload: no initial offset is subtracted.
            assert abs(tracking-value['cube_tracking_error_m'])<1e-12
            assert abs(orientation-value['cube_tracking_orientation_error_rad'])<1e-10
            assert abs(raw-value['raw_goal_position_error_m'])<1e-12
            assert abs(raw_angle-value['raw_goal_orientation_error_rad'])<1e-10
            assert abs(max(value['left_relative_slip_m'],value['right_relative_slip_m'])
                       -value['relative_slip_m'])<1e-12
            if t<4.:
                assert value['grasp_reference_established']==0 and value['relative_slip_m']==0
            else:
                assert value['grasp_reference_established']==1
            for key in peaks:
                peaks[key] = max(peaks[key],value[key])
            if t>=4. and row['phase'] not in ('unload','final_settle','final_hold'):
                active_grip_peak_slip = max(active_grip_peak_slip,value['relative_slip_m'])
            if first is None:
                first = dict(position=pos,desired=desired,quaternion=quat,
                             desired_quaternion=desired_quat,tracking=tracking)
            if row['phase']=='final_hold':
                final_count += 1
                assert tracking<.005 and raw<.01 and horizontal<.005 and raw_angle<.02
                assert value['cube_table_support_N']>=.8*report.get('cube_mass_kg',2.14)*report.get('gravity_m_s2',9.81)
                assert value['robot_table_load_N']<.5 and value['cube_speed_m_s']<.01 and q_error<.01
                assert all(value[key]==0 for key in ('squeeze_reference_N','support_x_per_hand_N',
                                                   'support_y_per_hand_N','support_z_per_hand_N'))
            if value['step_allowed']==0:
                assert row['phase']=='final_hold'
            last = dict(raw_goal_error_m=raw,horizontal_goal_error_m=horizontal,
                        goal_orientation_error_rad=raw_angle)
    assert n==applied+1 and applied==report['transport_physical_steps']
    assert settling['physical_steps']+applied==report['total_physical_steps']
    assert clipped==0 and final_count==1001
    assert peaks['obstacle_load_N']==peaks['robot_table_load_N']==0.
    assert first['tracking']<1e-12
    assert max(abs(first['desired'][i]-measured[i]) for i in range(3))<1e-12
    assert angle(first['desired_quaternion'],measured[3:])<1e-12
    assert abs(last['raw_goal_error_m']-report['final_raw_goal_position_error_m'])<1e-12
    assert abs(last['goal_orientation_error_rad']-report['final_goal_orientation_error_rad'])<1e-10
    assert abs(last['horizontal_goal_error_m']-report['final_horizontal_goal_error_m'])<1e-12
    assert abs(active_grip_peak_slip-report['peak_relative_slip_m'])<1e-12
    assert abs(peak_joint_error-report['peak_joint_error_rad'])<1e-12
    assert abs(peaks['cube_tracking_error_m']-report['peak_cube_tracking_error_m'])<1e-12
    assert abs(peaks['cube_tracking_orientation_error_rad']-report['peak_cube_orientation_error_rad'])<1e-10
    result = dict(independent_audit_passed=True,report=str(filename),rows=n,applied_steps=applied,
                  settling_physical_steps=settling['physical_steps'],first_observation=first,
                  final=last,peaks=peaks,active_grip_peak_relative_slip_m=active_grip_peak_slip,
                  peak_joint_error_rad=peak_joint_error,phases=phases,
                  reported_peaks_independently_verified=True,
                  clipped_steps=clipped,final_hold_samples=final_count,
                  direct_pose_errors_verified_every_phase=True,source_hashes_matched=True,
                  all_joint_limits_checked=True,quaternions_and_goal_orientation_checked=True)
    output = filename.with_name(filename.stem+'_audit.json')
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    print('Audit saved:',output)
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report',type=Path)
    audit(parser.parse_args().report)
