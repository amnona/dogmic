# dogmic
identify dog barks in dogmic or dogcam recording
also, calculate md5 for each movie and email (so timestamped)

installation:
1. setup a conda environment with loguru, shlex, librosa, dotenv
2. copy modify_me.env.txt to .env, and change relevant secrets
3. modify run_pipeline.bat to setup conda and python script positions
4. setup daily run - Configure Windows Task Scheduler 
    Search for Task Scheduler in the Windows Start menu and open it.
    In the right-hand pane, click Create Task...
    General Tab:
        Name: Give your task a meaningful name (e.g., Run Python Script Hourly).
        Security options: Select Run whether user is logged on or not so the script runs even if your computer is locked or you are signed out. You may need to enter your Windows password later.
        Check Run with highest privileges.
    Triggers Tab:
        Click New...
        Begin the task: Select On a schedule.
        Settings: Select Daily.
        Click OK.
    Actions Tab:
        Click New...
        Action: Ensure Start a program is selected.
        Program/script: Click Browse and select your created batch file (e.g., C:\Scripts\run_my_script.bat).
        Start in (optional): This should be the directory containing your batch file (e.g., C:\Scripts\).
        Click OK.
    Conditions/Settings Tabs: You can leave these as default unless you have specific power requirements (e.g., only run when on AC power).
    Click OK to save the task. Windows may prompt you to enter your user account password to save the credentials for "Run whether user is logged on or not". 
