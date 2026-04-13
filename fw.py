import argparse

from cli.commands import register_commands


class FrameworkHelpFormatter(argparse.HelpFormatter):
    def _format_action(self, action: argparse.Action) -> str:
        if isinstance(action, argparse._SubParsersAction):
            format_action = super()._format_action
            return self._join_parts(
                format_action(subaction)
                for subaction in self._iter_indented_subactions(action)
            )
        return super()._format_action(action)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='(HZNU OS Course Lab) Side Story: NN Image Filter',
        formatter_class=FrameworkHelpFormatter,
        usage='%(prog)s [-h] COMMAND ...',
    )
    subparsers = parser.add_subparsers(
        title='commands',
        dest='command',
        metavar='COMMAND',
        required=True,
    )
    register_commands(subparsers)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.command_handler(args)


if __name__ == '__main__':
    main()
