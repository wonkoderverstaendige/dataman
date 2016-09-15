import cmd
import logging


class DataManCLI(cmd.Cmd):
    """Command line tool for quick data documentation."""

    prompt = "dm> "
    intro = "Data Manager\n --Ronny's way of avoiding having to stare at spreadsheets."

    def __init__(self, completekey='tab', stdin=None, stdout=None):
        super().__init__(completekey='tab', stdin=None, stdout=None)
        self.log = logging.getLogger(__name__)

    def preloop(self):
        self.log.debug("starting DataMan CLI")
        # process command line arguments etc.

    def do_greet(self, user):
        """greet [user name]
        Simple user greeting. When used in combination with a parameter, will
        respond with personalized greeting. Yay."""
        if user:
            print("hello ", user)
        else:
            print("hi there!")

    def do_ls(self, path):
        if not len(path):
            path = '.'
        import dataman.lib.dirstats as ds
        ds.print_table(ds.gather(path))

    def do_stats(self, path):
        if not len(path):
            path = '.'
        import dataman.lib.dirstats as ds
        ds.print_table(ds.gather(path))

    def do_vis(self, path):
        from dataman.vis import vis
        vis.run(target=path)

    def do_exit(self, line):
        """Exit"""
        return True

    def do_EOF(self, line):
        """Exit"""
        return True

    def postloop(self):
        print("Done.")
