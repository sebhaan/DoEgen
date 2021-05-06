# Load settings

import argparse
import yaml

ap = argparse.ArgumentParser()
ap.add_argument('settings_path', nargs='?', default='settings_expresults.yaml')
args = ap.parse_args()
print(f"using settings in: {args.settings_path!r}")
with open(args.settings_path) as f:
	cfg = yaml.safe_load(f)
for key in cfg:
	locals()[str(key)] = cfg[key]