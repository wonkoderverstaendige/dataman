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
        if not path:
            path = "."
        tools.stats(path)
        table_hdr = "{0:^25}{sep}{1}{sep}{2}{sep}{3}{sep}{4}{sep}{5}{sep}{6}{sep}".format(
                "Folder name", "size", "#files", "#vid", "#img", "#snd", "format", sep="|")
        print table_hdr
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

