# Vis 
    [ ] Performance on mechanical harddrives sucks. 
    [ ] Dragging much slower than scrolling, probably updating constantly vs. once per buf_limit
    [ ] Zoom doesn't scale scrolling offset. Scrolling should be relative increments (stick to cursor)
    [ ] Parameter handling -> hand over the command line arguments to vis!
    [ ] Channel color scheme not working for single column (color channel groups from prb file)
    [ ] Doesn't seem to scroll the whole width (set_offset hardcorded limit at ~1M samples)
    [ ] Show timestamp somehow (how to text in OpenGL?)
    [ ] RightMB not taking current scale into account, starting "fresh"
    [ ] Switch to glumpy instead of vispy?
    
# General
    [ ] Integrate OIO
    [ ] Subcommand overhaul (oio+dm commands)
    [ ] dm ls should return number of datasets on subfolders
    [ ] Logging verbosity with --log=INFO etc.
    [ ] dataman configuration for quick loading/inspection/overrides (dataman.conf)
    
# Urgent
    [ ] Holy frick, we forgot about the streaming branch!
    