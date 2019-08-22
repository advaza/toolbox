# Copyright (c) 2017 Lightricks. All rights reserved.
"""
A wrapper for ArgumentParser class. It inherits from argparse.ArgumentParser with the following
extensions:

1. Ability to add parameters in a configuration file. If configuration file is given together with
command line parameters, the command line parameters override the config file parameters. The
configuration file can be a pickle/json/yaml or a txt ('.txt' is not necessary) file.
2. A singleton wrapper.
3. No need to call "parse_args" function explicitly. Parsing is done automatically when the first
param is called. For example,
                        args = Flags()
                        # ... add arguments ...
                        do_something_without_args() # arguments are not parsed yet
                        foo(args.param) # all arguments are being parsed
                        print(args.param2) # dictionary lookup only
                        bar(args.param) # dictionary lookup only
4. Parse arguments via a dictionary. This is useful when trying to use code from interactive
environment such as ipython/Jupyter. The dictionary arguments are override by the command line
arguments.

"""

import json
import pickle
import sys
import yaml


from argparse import ArgumentParser, Namespace


def process_dict(dict_path):
    """
    Parse a dictionary from dictionary serialization of pickle, yaml or json.
    :param dict_path: Path to serialized dictionary file which is with either pickle,
    yaml, or json.
    :return: Parsed label to name dictionary.
    """

    parsed_dict = try_pickle_parse(dict_path)
    if not parsed_dict:
        parsed_dict = try_yml_parse(dict_path)

    if not parsed_dict:
        parsed_dict = try_json_parse(dict_path)

    return parsed_dict


def try_pickle_parse(file_path):
    """
    Try to parse file `file_path` to a dictionary using Pickle.
    :param file_path: A string of a file path to parse.
    :return: A dict of the parsed file content if the file is a pickle file, otherwise None.
    """
    try:
        with open(file_path, 'rb') as pickle_file:
            parsed_dict = pickle.load(pickle_file)
    except (ValueError, TypeError, pickle.UnpicklingError):
        return None
    return parsed_dict


def try_json_parse(file_path):
    """
    Try to parse file `file_path` to a dictionary using Json.
    :param file_path: A string of a file path to parse.
    :return: A dict of the parsed file content if the file is a Jason file, otherwise None.
    """
    try:
        with open(file_path, 'r') as json_file:
            parsed_dict = json.loads(json_file)
    except (ValueError, TypeError):
        return None
    return parsed_dict


def try_yml_parse(file_path):
    """
    Try to parse file `file_path` to a dictionary using Yaml.
    :param file_path: A string of a file path to parse.
    :return: A dict of the parsed file content if the file is a yaml file, otherwise None.
    """
    try:
        with open(file_path, 'rb') as yaml_file:
            parsed_dict = yaml.load(yaml_file)
            if not isinstance(parsed_dict, dict):
                return None
    except (ValueError, TypeError, yaml.scanner.ScannerError):
        return None
    return parsed_dict


class MyArguments(ArgumentParser):

    def __init__(self, fromfile_prefix_chars=None):
        """
        Initializes extended argument parser. Initializes the `fromfile_prefix_chars` parameter to
        be the '@' character by default or user defining characters. To pass an argument file add a
        character from `fromfile_prefix_chars` at the beginning of the file path.
        For example: `python  my_script.py @/path/to/argument_file`

        :param fromfile_prefix_chars: A List or a Tuple of special characters that are used to
        indicate an argument file parameter. Default is [@].
        """
        if fromfile_prefix_chars is None:
            fromfile_prefix_chars = ["@"]

        self.args = None
        ArgumentParser.__init__(self, fromfile_prefix_chars=fromfile_prefix_chars)

        self.is_jupyter_notebook = any(["jupyter" in arg.lower() for arg in sys.argv[1:]])

        self.fromfile_prefix_chars = fromfile_prefix_chars

        # Initializes arguments dictionary.
        self.config_files_data = {}

    def __getattr__(self, name):
        """
        If arguments have not being parsed yet, parse them. Return attribute name using the
        parsed arguments dictionary.
        :param name: Argument name.
        :return:
        """

        if self.args is None:
            self.parse_args(args=None)

        return self.args.__dict__[name]

    def __setattr__(self, name, value):
        """
        Set value to argument
        :param name: Argument attribute.
        :param value: Value to assign.
        :return:
        """
        super(MyArguments, self).__setattr__(name, value)

        if self.args is not None and name in self.args.__dict__:
            self.args.__dict__[name] = value


    def _expand_arg_strings_from_file(self, config_file_path, new_arg_strings):
        """
        Adds line separated argument strings from a text file `config_file_path` to the argument
         strings list `new_arg_strings`.
        :param config_file_path: A file path of line separated argument strings.
        :param new_arg_strings: Argument strings list that will be expanded from file arguments.
        :return:
        """

        with open(config_file_path) as config_file:
            try:
                arg_strings = []
                for arg_line in config_file.read().splitlines():
                    for arg in self.convert_arg_line_to_args(arg_line):
                        arg_strings.append(arg)
                arg_strings = self._read_args_from_files(arg_strings)
                new_arg_strings.extend(arg_strings)
            except OSError:
                err = sys.exc_info()[1]
                self.error(str(err))

    def _read_args_from_files(self, arg_strings):
        """
        Override function.
        The parent function parses arguments from a simple text file that contains line separated
        arguments. This function extends the parent function by enabling argument file to be a
        dictionary-like file: json, pickle or yaml file.
        :param arg_strings: A list of argument strings collected so far, this list is extended
        recursively with arguments from files.
        :return: new_arg_strings: The extended list of arguments from arguments files.
        """
        # Expand arguments referencing files
        new_arg_strings = []
        for arg_string in arg_strings:

            # For regular arguments, just add them back into the list
            if not arg_string or arg_string[0] not in self.fromfile_prefix_chars:
                new_arg_strings.append(arg_string)

            # Process arguments from arguments file. If an arguments file is a dict-like file then
            # parse the argument to produce arguments dictionary. If it is a text file, add the
            # argument strings from the file to the argument strings list of the command line.
            else:
                config_file_path = arg_string[1:]
                config_file_args = process_dict(dict_path=config_file_path)

                if config_file_args:
                    self.config_files_data.update(config_file_args)

                else:
                    self._expand_arg_strings_from_file(
                        config_file_path=config_file_path,
                        new_arg_strings=new_arg_strings
                    )

        # Return the modified argument list that will be parsed using the parent argument string
        # parser.
        return new_arg_strings

    def process_default_namespace(self, namespace_or_dict_args):
        """
        Builds a default namespace that will be used as the new default values of the arguments
        before parsing the command line arguments. Initializes arguments dict using the dict-like
        argument file(s) that are in `self.config_files_data`. Then, update or add arguments
        using either `namespace` dict or `dict_args`. The dictionary cannot be initialize
        using both namespace and dictionary argument since it is
        :param namespace: argparse.Namespace to initialize argument values.
        :param dict_args: arguments dict to initialize argument values.
        :return: argparse.Namespace with the initialized argument values.
        """

        if namespace_or_dict_args:
            if isinstance(namespace_or_dict_args, Namespace):
                self.config_files_data.update(namespace_or_dict_args._get_kwarg)
            elif isinstance(namespace_or_dict_args, dict):
                self.config_files_data.update(namespace_or_dict_args)
            else:
                raise TypeError('namespace_or_dict_args must be either a namespace or a dict, not '
                                '{!r}'.format(namespace_or_dict_args.__class__.__name__))

        if self.config_files_data:
            return Namespace(**self.config_files_data)
        return None

    def parse_args(self, args=None, namespace_or_dict_args=None):
        """
        Parse arguments from arguments list or command line arguments. Set default values to values
        in `namespace_or_dict_args` if given, before parsing.
        :param args: Iterable of strings that should be parsed. If None, initialized to command
        line arguments - sys.argv.
        :param namespace_or_dict_args: Namespace or a dictionary or arguments that should be set as
        default values.
        :return: Arguments namespace, instance of argparse.Namespace.
        """

        if args is None:
            if self.is_jupyter_notebook:
                args = []
            else:
                args = sys.argv[1:]

        config_files_args = [arg for arg in args if arg[0] in self.fromfile_prefix_chars]
        other_args = [arg for arg in args if arg[0] not in self.fromfile_prefix_chars]

        # Parse the config files arguments and update the dict args so that arguments values are
        # updated in the order of their appearance.
        for config_file in config_files_args:
            ArgumentParser.parse_args(self, args=[config_file])

        # Process the default arguments that override the default arguments values (that are
        # defined upon arguments declaration). These are the config files arguments, override by
        # user define namespace or dictionary (use either namespace or dict_args, can't use both).
        default_namespace = self.process_default_namespace(namespace_or_dict_args)

        # The new default arguments override by command line arguments.
        self.args = ArgumentParser.parse_args(self, args=other_args, namespace=default_namespace)

        return self.args

    def get_argument_dict(self):
        """
        Returns the parsed arguments as dict.
        :return: Dict of the parsed arguments.
        """
        if self.args is None:
            self.parse_args(args=None)

        return self.args.__dict__


class SingletonDecorator:
    """
    A Singleton decorator for a class `klass`.
    """

    def __init__(self, klass):
        self.klass = klass
        self.instance = None

    def __call__(self, *args, **kwds):
        if self.instance is None:
            self.instance = self.klass(*args, **kwds)
        return self.instance


Flags = SingletonDecorator(MyArguments)
