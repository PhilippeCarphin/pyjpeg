import argparse
import json


def load_config_file(filename):
    try:
        with open(filename) as f:
            config = json.loads(f.read())
            return config
    except FileNotFoundError:
        return {}


def command_line_parser():
    p = argparse.ArgumentParser()
    p.add_argument("-v", "--verbose", action="store_true", help="Address of the server")
    p.add_argument("-i", "--input", help='input file for JPEG compression')
    p.add_argument("-o", "--output", help='output file for JPEG compression')
    p.add_argument("-c", "--create-config-file", help='create a config file with for this program')
    return p


def create_config_file(program_options):
    """ I could probably use the parser to construct this
    dictionnary """
    config = {
        'verbose': True,
        'input': './input_image.png',
        'output': './output_image.png'
    }
    config = program_options.__dict__
    filename = program_options.create_config_file
    with open(filename, 'w+') as f:
        f.write(json.dumps(config, indent=4, sort_keys=True))


def post_parsing(options):
    # if no input, use our test image

    # Ex: Check if file arguments exist,

    config = load_config_file('./.pyjpeg')
    for k in config:
        if k != '__len__':
            options.__dict__[k] = config[k]

    # Defaults
    #    Default to environment variables
    #    Default to values in a config file
    # Command line options override everything
    # Config file overrides environment
    # Environment is last

    return options


def get_options():
    parsed_options = command_line_parser().parse_args()

    program_options = post_parsing(parsed_options)

    d = parsed_options.__dict__
    if program_options.create_config_file:
        create_config_file(program_options)

    return program_options


if __name__ == "__main__":
    get_options()
