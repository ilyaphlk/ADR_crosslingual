import yaml

with open('example_config.yaml') as cfg:
	d = yaml.load(cfg, Loader=yaml.FullLoader)
	print(d)