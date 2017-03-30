# Vis 
    [ ] Performance on mechanical harddrives sucks. 
    [ ] Dragging much slower than scrolling, probably updating constantly vs. once per buf_limit
    [ ] Zoom doesn't scale scrolling offset. Scrolling should be relative increments (stick to cursor)
    [X] Parameter handling -> hand over the command line arguments to vis!
    [X] Channel color scheme not working for single column (color channel groups from prb file)
    [X] Doesn't seem to scroll the whole width (set_offset hardcorded limit at ~1M samples)
    [.] Show timestamp somehow (how to text in OpenGL?)
    [ ] RightMB not taking current scale into account, starting "fresh"
    [ ] Switch to glumpy instead of vispy?
    [X] Allow .dat files as input to vis
    [ ] Non-filling channel number
    [X] Allow single channel
    [X] Rescale vertical size when few channels shown (have margin pressure)
    [ ] Shared state Values with Streamer, update on full data load
    [ ] First chunk sometimes displayed before loaded -> update procedure not synchronized enough
    
# General
    [X] Integrate OIO
    [X] Subcommand overhaul (oio+dm commands)
    [ ] dm ls should return number of datasets on subfolders
    [ ] Logging verbosity with --log=INFO etc.
    [ ] dataman configuration for quick loading/inspection/overrides (dataman.conf)
    [X] Update setup.py with literate syntax to allow requirement scraping by IDE
    [ ] Spaces in filenames
    [ ] dead_channels -> bad_channels

# LS/Stats
    [ ] Order file list by file/directory name!
    
# Urgent
    [X] Holy frick, we forgot about the streaming branch!
    
# Ref
    [X] Make ref file
    [X] Subtract ref file
    [ ] Ref using prb file of same name in same dir -L

# Split
    [X] Split based on groupings
    [X] Split based on layout file (channel groups only)
    