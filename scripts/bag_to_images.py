#!/usr/bin/env python3
import rosbag, cv2, os, argparse
from cv_bridge import CvBridge

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Path to dataset (.bag files)")
parser.add_argument("--output", required=True, help="Path to save frames")
args = parser.parse_args()

bridge = CvBridge()

for school in os.listdir(args.input):
    school_path = os.path.join(args.input, school)
    if not os.path.isdir(school_path):
        continue

    for bagfile in os.listdir(school_path):
        if not bagfile.endswith(".bag"):
            continue

        bag_path = os.path.join(school_path, bagfile)

        # e.g. AU_chunk01.bag -> 01
        base = os.path.splitext(bagfile)[0]
        if "chunk" in base:
            num = base.split("chunk")[-1]
        else:
            num = base

        save_path = os.path.join(args.output, school, num)
        os.makedirs(save_path, exist_ok=True)

        print(f"[INFO] Processing {bag_path} -> {save_path}")

        bag = rosbag.Bag(bag_path)
        count = 0
        for topic, msg, t in bag.read_messages(topics=['/rail_robot/pano_image/compressed']):
            img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            cv2.imwrite(os.path.join(save_path, f"frame_{count:06d}.jpg"), img)
            count += 1
        bag.close()

        print(f"[INFO] Saved {count} frames for {school}/{num}")
