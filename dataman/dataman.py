#!/usr/bin/env python

# ? is short for builtin help
# ! allows shelling out

import sys
import cmd
import tools

class DataMan(cmd.Cmd):
    """Command line tool for quick data documentation."""

    prompt = "dm> "
    intro = "Data Manager\n --Ronny's way of avoiding having to stare at spreadsheets."

    def preloop(self):
        pass
        # process command line arguments etc.

    def do_greet(self, user):
        """greet [user name]
        Simple user greeting. When used in combination with a parameter, will
        respond with personalized greeting. Yay."""
        if user:
            print "hello ", user
        else:
            print "hi there!"

    def do_stats(self, path):
        if not len(path):
            path = '.'
        import folderstats as fs
        fs.print_table(fs.gather(path))

    def do_EOF(self, line):
        "Exit"
        return True

    def postloop(self):
        print "Done."

if __name__ == "__main__":
    if len(sys.argv) > 1:
        DataMan().onecmd(' '.join(sys.argv[1:]))
    else:
        DataMan().cmdloop()

