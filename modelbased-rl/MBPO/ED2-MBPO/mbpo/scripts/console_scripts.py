import click

@click.group(invoke_without_command=True)
@click.argument("data_path", required=True, nargs=1)
@click.argument('kwargs', nargs=-1)
def cli(data_path, kwargs):
    from viskit.frontend import main
    main()


def main():
    return cli()

if __name__ == "__main__":
    main()
