#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2026.1.2),
    on novembro 05, 2026, at 14:46
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (
    NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, STOPPING, FINISHED, PRESSED, 
    RELEASED, FOREVER, priority
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard
from psychopy.hardware import camera
from psychopy.sound import microphone

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2026.1.2'
expName = 'TrainingTask_PreTest'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'expVersion|hid': expVersion,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = (1024, 768)
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # replace default participant ID
    if prefs.piloting['replaceParticipantID']:
        expInfo['participant'] = 'pilot'

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\Asus\\Documents\\recursos\\H-IBL.Task\\Final_Task_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    # store pilot mode in data file
    thisExp.addData('piloting', PILOTING, priority=priority.LOW)
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=1,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    if PILOTING:
        # show a visual indicator if we're in piloting mode
        if prefs.piloting['showPilotingIndicator']:
            win.showPilotingIndicator()
        # always show the mouse in piloting mode
        if prefs.piloting['forceMouseVisible']:
            win.mouseVisible = True
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ptb'
        )
    # initialize 'WebCam'
    deviceManager.addDevice(
        deviceName='WebCam',
        deviceClass='psychopy.hardware.camera.CameraDevice',
        device=49,
        captureLib='ffpyplayer',
        pixelFormat='yuyv422',
        codecFormat='',
        captureAPI='DirectShow',
        frameRate=None,
        frameSize=[]
    )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    currentRoutine : psychopy.data.Routine
        Current Routine we are in at time of pausing, if any. This object tells PsychoPy what Components to pause/play/dispatch.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='PsychToolbox',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # dispatch messages on response components
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents():
                comp.device.dispatchMessages()
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # update experiment info
    expInfo['date'] = data.getDateStr()
    expInfo['expName'] = expName
    expInfo['expVersion'] = expVersion
    expInfo['psychopyVersion'] = psychopyVersion
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='PsychToolbox'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "WelcomeScreen" ---
    # Run 'Begin Experiment' code from GlobalCam
    import cv2, threading, csv, atexit
    from psychopy import core
    
    CAM_INDEX = 1                       # 0 = built-in, 1 = external USB C270
    REQ_W, REQ_H, REQ_FPS = 640, 480, 30
    EXPOSURE = -6                       # try -5, -6, -7 (more negative = darker but higher fps)
    
    class CamRecorder:
        def __init__(self, index, video_path, csv_path):
            self.ok = False
            self.n = 0              # real frames captured
            self.written = 0        # frames written to the video file
            self.fps = float(REQ_FPS)
            self._stop = threading.Event()
            self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQ_W)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQ_H)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FPS, REQ_FPS)
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)   # manual exposure mode
            self.cap.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE)
            if not (self.cap.isOpened() and self.cap.read()[0]):
                print("Camera failed to open. Running without recording.")
                return
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
            fmt = "".join(chr((fc >> 8 * i) & 0xFF) for i in range(4))
            self.writer = cv2.VideoWriter(
                video_path, cv2.VideoWriter_fourcc(*"MJPG"), self.fps, (w, h))
            self._csvfile = open(csv_path, "w", newline="")
            self._csv = csv.writer(self._csvfile)
            self._csv.writerow(["real_frame", "video_frame", "psychopy_time"])
            self.t_first = None
            self.t_last = None
            self.ok = True
            print(f"Camera OK: {w}x{h}, format {fmt}, file written at {self.fps} fps")
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()
    
        def _loop(self):
            t0 = None
            while not self._stop.is_set():
                ok, frame = self.cap.read()
                t = core.getTime()
                if not ok:
                    continue
                if t0 is None:
                    t0 = t
                    self.t_first = t
                self.t_last = t
                # number of frames the file should contain at this instant
                target = int((t - t0) * self.fps) + 1
                first_idx = self.written
                while self.written < target:
                    self.writer.write(frame)
                    self.written += 1
                # real frame -> index of the first video-file frame that represents it
                self._csv.writerow([self.n, first_idx, t])
                self.n += 1
    
        def stop(self):
            if not self.ok:
                return
            self.ok = False
            self._stop.set()
            self.thread.join(timeout=2)
            self.writer.release()
            self.cap.release()
            self._csvfile.close()
            if self.t_last and self.t_first and self.t_last > self.t_first:
                dur = self.t_last - self.t_first
                print(f"Actual camera fps: {self.n / dur:.1f} "
                      f"({self.n} real frames, {self.written} in file, {dur:.1f} s)")
    
    try:
        base = filename
    except NameError:
        base = thisExp.dataFileName
    
    video_path = base + "_cam.avi"
    csv_path = base + "_cam_timestamps.csv"
    rec = CamRecorder(CAM_INDEX, video_path, csv_path)
    atexit.register(rec.stop)
    textWelcome = visual.TextStim(win=win, name='textWelcome',
        text='Irá iniciar a tarefa. A mesma consiste na apresentação de um contraste à direita/esquerda. Indique em cada trial o lado do estímulo utilizando a tecla -S- para o lado esquerdo e  -L- para o lado direito. \n\nPedimos que mantenha o olhar no centro do ecrã durante a experiência\n\nPressione a SPACEBAR para começar',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_respWelcome = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "blank2000" ---
    textBlankInst = visual.TextStim(win=win, name='textBlankInst',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "InstructionsTutorial" ---
    textTutorial = visual.TextStim(win=win, name='textTutorial',
        text='Irá realizar uma pequena sessão de treino.\n\nQuando o estímulo aparecer do lado direito pressione -L- e quando aparecer do lado esquerdo pressione -S-\n\nEstas respostas não serão contabilizadas.\n\nPressione SPACEBAR para iniciar',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_tutorial = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "Tutorial" ---
    # set audio backend
    sound.Sound.backend = 'ptb'
    soundTutGoCue = sound.Sound(
        'A', 
        secs=0.1, 
        stereo=True, 
        hamming=True, 
        speaker=None,    name='soundTutGoCue'
    )
    soundTutGoCue.setVolume(1.0)
    polygonHorizontalTut = visual.Rect(
        win=win, name='polygonHorizontalTut',units='deg', 
        width=[1.0, 1.0][0], height=[1.0, 1.0][1],
        ori=1.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=(0.0000, 0.0000, 0.0000), fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    polygonVerticalTut = visual.Rect(
        win=win, name='polygonVerticalTut',units='deg', 
        width=[1.0, 1.0][0], height=[1.0, 1.0][1],
        ori=1.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=(0.0000, 0.0000, 0.0000), fillColor='white',
        opacity=None, depth=-2.0, interpolate=True)
    gaborTut = visual.GratingStim(
        win=win, name='gaborTut',units='deg', 
        tex=None, mask='sin', anchor='center',
        ori=1.0, pos=[0,0], draggable=False, size=1.0, sf=1.0, phase=1.0,
        color=[1,1,1], colorSpace='rgb',
        opacity=None, contrast=1.0, blendmode='avg',
        texRes=256.0, interpolate=True, depth=-3.0)
    # Run 'Begin Experiment' code from codeMoveT
    from psychopy.hardware import keyboard as psy_keyboard
    
    kbd = psy_keyboard.Keyboard()
    
    
    # --- Initialize components for Routine "TutFeedback" ---
    # Run 'Begin Experiment' code from codeFcondTut
    scoreT = 0
    textFBTut = visual.TextStim(win=win, name='textFBTut',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    soundFBTut = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker=None,    name='soundFBTut'
    )
    soundFBTut.setVolume(1.0)
    
    # --- Initialize components for Routine "Blank4000" ---
    textBlank = visual.TextStim(win=win, name='textBlank',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "InstructionsFullTask" ---
    textFullTask = visual.TextStim(win=win, name='textFullTask',
        text='Irá iniciar a tarefa completa.\n\nQuando o estímulo aparecer do lado direito pressione -L- e quando aparecer do lado esquerdo pressione -S-\n\nPressione SPACEBAR para iniciar',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    keyRespFullTask = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "FullTask" ---
    soundGoCue = sound.Sound(
        'A', 
        secs=0.1, 
        stereo=True, 
        hamming=True, 
        speaker=None,    name='soundGoCue'
    )
    soundGoCue.setVolume(1.0)
    polygonVertical = visual.Rect(
        win=win, name='polygonVertical',units='deg', 
        width=[1.0, 1.0][0], height=[1.0, 1.0][1],
        ori=1.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=(0.0000, 0.0000, 0.0000), fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    polygonHorizontal = visual.Rect(
        win=win, name='polygonHorizontal',units='deg', 
        width=[1.0, 1.0][0], height=[1.0, 1.0][1],
        ori=1.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=(0.0000, 0.0000, 0.0000), fillColor='white',
        opacity=None, depth=-2.0, interpolate=True)
    gabor = visual.GratingStim(
        win=win, name='gabor',units='deg', 
        tex=None, mask='sin', anchor='center',
        ori=1.0, pos=[0,0], draggable=False, size=1.0, sf=1.0, phase=1.0,
        color=[1,1,1], colorSpace='rgb',
        opacity=None, contrast=1.0, blendmode='avg',
        texRes=256.0, interpolate=True, depth=-3.0)
    # Run 'Begin Experiment' code from codeCorKey
    total_task_timer = core.Clock()
    
    # Run 'Begin Experiment' code from codeMove
    from psychopy.hardware import keyboard as psy_keyboard
    
    kbd = psy_keyboard.Keyboard()
    
    
    # --- Initialize components for Routine "FullTaskFeedback" ---
    # Run 'Begin Experiment' code from codeFconditions
    score = 0
    textFeedback = visual.TextStim(win=win, name='textFeedback',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    soundFeedback = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker=None,    name='soundFeedback'
    )
    soundFeedback.setVolume(1.0)
    
    # --- Initialize components for Routine "Pause" ---
    textPause = visual.TextStim(win=win, name='textPause',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    Returntotasktext = visual.TextStim(win=win, name='Returntotasktext',
        text='Carrega na SPACEBAR para recomeçar',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    key_resp = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "Blank4000" ---
    textBlank = visual.TextStim(win=win, name='textBlank',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "GoodbyeScreen" ---
    textEnd = visual.TextStim(win=win, name='textEnd',
        text='Chegou ao fim da tarefa. \n\nEnviaremos os leaderboard da sessão por email. Constará apenas o número de ID, garantido o seu anonimato. Caso tenha interesse em participar pressione -y-, caso contrário pressione -n-.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    keyLeaderBoard = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "IDScreen" ---
    textGoodbye = visual.TextStim(win=win, name='textGoodbye',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    if eyetracker is not None:
        eyetracker.enableEventReporting()
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "WelcomeScreen" ---
    # create an object to store info about Routine WelcomeScreen
    WelcomeScreen = data.Routine(
        name='WelcomeScreen',
        components=[textWelcome, key_respWelcome],
    )
    WelcomeScreen.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_respWelcome
    key_respWelcome.keys = []
    key_respWelcome.rt = []
    _key_respWelcome_allKeys = []
    # store start times for WelcomeScreen
    WelcomeScreen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    WelcomeScreen.tStart = globalClock.getTime(format='float')
    WelcomeScreen.status = STARTED
    thisExp.addData('WelcomeScreen.started', WelcomeScreen.tStart)
    WelcomeScreen.maxDuration = None
    # keep track of which components have finished
    WelcomeScreenComponents = WelcomeScreen.components
    for thisComponent in WelcomeScreen.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "WelcomeScreen" ---
    thisExp.currentRoutine = WelcomeScreen
    WelcomeScreen.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # Run 'Each Frame' code from GlobalCam
        thisExp.addData("cam_frame_at_stim", rec.n)
        thisExp.addData("cam_time_at_stim", core.getTime())
        
        # *textWelcome* updates
        
        # if textWelcome is starting this frame...
        if textWelcome.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            textWelcome.frameNStart = frameN  # exact frame index
            textWelcome.tStart = t  # local t and not account for scr refresh
            textWelcome.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(textWelcome, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'textWelcome.started')
            # update status
            textWelcome.status = STARTED
            textWelcome.setAutoDraw(True)
        
        # if textWelcome is active this frame...
        if textWelcome.status == STARTED:
            # update params
            pass
        
        # *key_respWelcome* updates
        waitOnFlip = False
        
        # if key_respWelcome is starting this frame...
        if key_respWelcome.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_respWelcome.frameNStart = frameN  # exact frame index
            key_respWelcome.tStart = t  # local t and not account for scr refresh
            key_respWelcome.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_respWelcome, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_respWelcome.started')
            # update status
            key_respWelcome.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_respWelcome.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_respWelcome.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_respWelcome.status == STARTED and not waitOnFlip:
            theseKeys = key_respWelcome.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_respWelcome_allKeys.extend(theseKeys)
            if len(_key_respWelcome_allKeys):
                key_respWelcome.keys = _key_respWelcome_allKeys[-1].name  # just the last key pressed
                key_respWelcome.rt = _key_respWelcome_allKeys[-1].rt
                key_respWelcome.duration = _key_respWelcome_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=WelcomeScreen,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            WelcomeScreen.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if WelcomeScreen.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in WelcomeScreen.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "WelcomeScreen" ---
    for thisComponent in WelcomeScreen.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for WelcomeScreen
    WelcomeScreen.tStop = globalClock.getTime(format='float')
    WelcomeScreen.tStopRefresh = tThisFlipGlobal
    thisExp.addData('WelcomeScreen.stopped', WelcomeScreen.tStop)
    # check responses
    if key_respWelcome.keys in ['', [], None]:  # No response was made
        key_respWelcome.keys = None
    thisExp.addData('key_respWelcome.keys',key_respWelcome.keys)
    if key_respWelcome.keys != None:  # we had a response
        thisExp.addData('key_respWelcome.rt', key_respWelcome.rt)
        thisExp.addData('key_respWelcome.duration', key_respWelcome.duration)
    thisExp.nextEntry()
    # the Routine "WelcomeScreen" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "blank2000" ---
    # create an object to store info about Routine blank2000
    blank2000 = data.Routine(
        name='blank2000',
        components=[textBlankInst],
    )
    blank2000.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for blank2000
    blank2000.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    blank2000.tStart = globalClock.getTime(format='float')
    blank2000.status = STARTED
    thisExp.addData('blank2000.started', blank2000.tStart)
    blank2000.maxDuration = None
    # keep track of which components have finished
    blank2000Components = blank2000.components
    for thisComponent in blank2000.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "blank2000" ---
    thisExp.currentRoutine = blank2000
    blank2000.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 2.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *textBlankInst* updates
        
        # if textBlankInst is starting this frame...
        if textBlankInst.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            textBlankInst.frameNStart = frameN  # exact frame index
            textBlankInst.tStart = t  # local t and not account for scr refresh
            textBlankInst.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(textBlankInst, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'textBlankInst.started')
            # update status
            textBlankInst.status = STARTED
            textBlankInst.setAutoDraw(True)
        
        # if textBlankInst is active this frame...
        if textBlankInst.status == STARTED:
            # update params
            pass
        
        # if textBlankInst is stopping this frame...
        if textBlankInst.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > textBlankInst.tStartRefresh + 2-frameTolerance:
                # keep track of stop time/frame for later
                textBlankInst.tStop = t  # not accounting for scr refresh
                textBlankInst.tStopRefresh = tThisFlipGlobal  # on global time
                textBlankInst.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textBlankInst.stopped')
                # update status
                textBlankInst.status = FINISHED
                textBlankInst.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=blank2000,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            blank2000.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if blank2000.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in blank2000.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "blank2000" ---
    for thisComponent in blank2000.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for blank2000
    blank2000.tStop = globalClock.getTime(format='float')
    blank2000.tStopRefresh = tThisFlipGlobal
    thisExp.addData('blank2000.stopped', blank2000.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if blank2000.maxDurationReached:
        routineTimer.addTime(-blank2000.maxDuration)
    elif blank2000.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-2.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "InstructionsTutorial" ---
    # create an object to store info about Routine InstructionsTutorial
    InstructionsTutorial = data.Routine(
        name='InstructionsTutorial',
        components=[textTutorial, key_resp_tutorial],
    )
    InstructionsTutorial.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_tutorial
    key_resp_tutorial.keys = []
    key_resp_tutorial.rt = []
    _key_resp_tutorial_allKeys = []
    # store start times for InstructionsTutorial
    InstructionsTutorial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    InstructionsTutorial.tStart = globalClock.getTime(format='float')
    InstructionsTutorial.status = STARTED
    thisExp.addData('InstructionsTutorial.started', InstructionsTutorial.tStart)
    InstructionsTutorial.maxDuration = None
    # keep track of which components have finished
    InstructionsTutorialComponents = InstructionsTutorial.components
    for thisComponent in InstructionsTutorial.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "InstructionsTutorial" ---
    thisExp.currentRoutine = InstructionsTutorial
    InstructionsTutorial.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *textTutorial* updates
        
        # if textTutorial is starting this frame...
        if textTutorial.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            textTutorial.frameNStart = frameN  # exact frame index
            textTutorial.tStart = t  # local t and not account for scr refresh
            textTutorial.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(textTutorial, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'textTutorial.started')
            # update status
            textTutorial.status = STARTED
            textTutorial.setAutoDraw(True)
        
        # if textTutorial is active this frame...
        if textTutorial.status == STARTED:
            # update params
            pass
        
        # *key_resp_tutorial* updates
        waitOnFlip = False
        
        # if key_resp_tutorial is starting this frame...
        if key_resp_tutorial.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_tutorial.frameNStart = frameN  # exact frame index
            key_resp_tutorial.tStart = t  # local t and not account for scr refresh
            key_resp_tutorial.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_tutorial, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_tutorial.started')
            # update status
            key_resp_tutorial.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_tutorial.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_tutorial.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_tutorial.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_tutorial.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_tutorial_allKeys.extend(theseKeys)
            if len(_key_resp_tutorial_allKeys):
                key_resp_tutorial.keys = _key_resp_tutorial_allKeys[-1].name  # just the last key pressed
                key_resp_tutorial.rt = _key_resp_tutorial_allKeys[-1].rt
                key_resp_tutorial.duration = _key_resp_tutorial_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=InstructionsTutorial,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            InstructionsTutorial.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if InstructionsTutorial.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in InstructionsTutorial.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "InstructionsTutorial" ---
    for thisComponent in InstructionsTutorial.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for InstructionsTutorial
    InstructionsTutorial.tStop = globalClock.getTime(format='float')
    InstructionsTutorial.tStopRefresh = tThisFlipGlobal
    thisExp.addData('InstructionsTutorial.stopped', InstructionsTutorial.tStop)
    # check responses
    if key_resp_tutorial.keys in ['', [], None]:  # No response was made
        key_resp_tutorial.keys = None
    thisExp.addData('key_resp_tutorial.keys',key_resp_tutorial.keys)
    if key_resp_tutorial.keys != None:  # we had a response
        thisExp.addData('key_resp_tutorial.rt', key_resp_tutorial.rt)
        thisExp.addData('key_resp_tutorial.duration', key_resp_tutorial.duration)
    thisExp.nextEntry()
    # the Routine "InstructionsTutorial" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    PreTest1TrialsTut = data.TrialHandler2(
        name='PreTest1TrialsTut',
        nReps=3, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('Conditions_Tut.xlsx'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(PreTest1TrialsTut)  # add the loop to the experiment
    thisPreTest1TrialsTut = PreTest1TrialsTut.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPreTest1TrialsTut.rgb)
    if thisPreTest1TrialsTut != None:
        for paramName in thisPreTest1TrialsTut:
            globals()[paramName] = thisPreTest1TrialsTut[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisPreTest1TrialsTut in PreTest1TrialsTut:
        PreTest1TrialsTut.status = STARTED
        if hasattr(thisPreTest1TrialsTut, 'status'):
            thisPreTest1TrialsTut.status = STARTED
        currentLoop = PreTest1TrialsTut
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisPreTest1TrialsTut.rgb)
        if thisPreTest1TrialsTut != None:
            for paramName in thisPreTest1TrialsTut:
                globals()[paramName] = thisPreTest1TrialsTut[paramName]
        
        # --- Prepare to start Routine "Tutorial" ---
        # create an object to store info about Routine Tutorial
        Tutorial = data.Routine(
            name='Tutorial',
            components=[soundTutGoCue, polygonHorizontalTut, polygonVerticalTut, gaborTut],
        )
        Tutorial.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        soundTutGoCue.setSound('5000', secs=0.1, hamming=True)
        soundTutGoCue.setVolume(1.0, log=False)
        soundTutGoCue.seek(0)
        gaborTut.setContrast(gratingT)
        # Run 'Begin Routine' code from codePhaseT
        # Generate random phase
        current_phaseT = np.random.uniform(0, 2 * np.pi)
        
        gaborTut.phase = current_phaseT
        
        # Creates a gabor phase column on the csv
        thisExp.addData('gabor_phaseT', current_phaseT)
        
        # Run 'Begin Routine' code from codeMoveT
        try:
        
            if hasattr(positionT[0], '__iter__') or isinstance(position[0], (list, tuple)):
                pos_x = float(positionT[0][0])
                pos_y = float(positionT[0][1])
            else:
                pos_x = float(positionT[0])
                pos_y = float(positionT[1])
        except Exception:
            pos_x = 0.0
            pos_y = 0.0
        
        
        kbd.clearEvents()
        
        # store start times for Tutorial
        Tutorial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Tutorial.tStart = globalClock.getTime(format='float')
        Tutorial.status = STARTED
        thisExp.addData('Tutorial.started', Tutorial.tStart)
        Tutorial.maxDuration = None
        # keep track of which components have finished
        TutorialComponents = Tutorial.components
        for thisComponent in Tutorial.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Tutorial" ---
        thisExp.currentRoutine = Tutorial
        Tutorial.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 11.0:
            # if trial has changed, end Routine now
            if hasattr(thisPreTest1TrialsTut, 'status') and thisPreTest1TrialsTut.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *soundTutGoCue* updates
            
            # if soundTutGoCue is starting this frame...
            if soundTutGoCue.status == NOT_STARTED and tThisFlip >= 0.1-frameTolerance:
                # keep track of start time/frame for later
                soundTutGoCue.frameNStart = frameN  # exact frame index
                soundTutGoCue.tStart = t  # local t and not account for scr refresh
                soundTutGoCue.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('soundTutGoCue.started', tThisFlipGlobal)
                # update status
                soundTutGoCue.status = STARTED
                soundTutGoCue.play(when=win)  # sync with win flip
            
            # if soundTutGoCue is stopping this frame...
            if soundTutGoCue.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > soundTutGoCue.tStartRefresh + 0.1-frameTolerance or soundTutGoCue.isFinished:
                    # keep track of stop time/frame for later
                    soundTutGoCue.tStop = t  # not accounting for scr refresh
                    soundTutGoCue.tStopRefresh = tThisFlipGlobal  # on global time
                    soundTutGoCue.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'soundTutGoCue.stopped')
                    # update status
                    soundTutGoCue.status = FINISHED
                    soundTutGoCue.stop()
            
            # *polygonHorizontalTut* updates
            
            # if polygonHorizontalTut is starting this frame...
            if polygonHorizontalTut.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                polygonHorizontalTut.frameNStart = frameN  # exact frame index
                polygonHorizontalTut.tStart = t  # local t and not account for scr refresh
                polygonHorizontalTut.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(polygonHorizontalTut, 'tStartRefresh')  # time at next scr refresh
                # update status
                polygonHorizontalTut.status = STARTED
                polygonHorizontalTut.setAutoDraw(True)
            
            # if polygonHorizontalTut is active this frame...
            if polygonHorizontalTut.status == STARTED:
                # update params
                polygonHorizontalTut.setPos(fixation_posT, log=False)
                polygonHorizontalTut.setSize(fixation_sizeT, log=False)
                polygonHorizontalTut.setOri(180.0, log=False)
                polygonHorizontalTut.setLineWidth(fixation_widthT, log=False)
            
            # if polygonHorizontalTut is stopping this frame...
            if polygonHorizontalTut.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > polygonHorizontalTut.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    polygonHorizontalTut.tStop = t  # not accounting for scr refresh
                    polygonHorizontalTut.tStopRefresh = tThisFlipGlobal  # on global time
                    polygonHorizontalTut.frameNStop = frameN  # exact frame index
                    # update status
                    polygonHorizontalTut.status = FINISHED
                    polygonHorizontalTut.setAutoDraw(False)
            
            # *polygonVerticalTut* updates
            
            # if polygonVerticalTut is starting this frame...
            if polygonVerticalTut.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                polygonVerticalTut.frameNStart = frameN  # exact frame index
                polygonVerticalTut.tStart = t  # local t and not account for scr refresh
                polygonVerticalTut.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(polygonVerticalTut, 'tStartRefresh')  # time at next scr refresh
                # update status
                polygonVerticalTut.status = STARTED
                polygonVerticalTut.setAutoDraw(True)
            
            # if polygonVerticalTut is active this frame...
            if polygonVerticalTut.status == STARTED:
                # update params
                polygonVerticalTut.setPos(fixation_posT, log=False)
                polygonVerticalTut.setSize(fixation_sizeT, log=False)
                polygonVerticalTut.setOri(90.0, log=False)
                polygonVerticalTut.setLineWidth(fixation_widthT, log=False)
            
            # if polygonVerticalTut is stopping this frame...
            if polygonVerticalTut.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > polygonVerticalTut.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    polygonVerticalTut.tStop = t  # not accounting for scr refresh
                    polygonVerticalTut.tStopRefresh = tThisFlipGlobal  # on global time
                    polygonVerticalTut.frameNStop = frameN  # exact frame index
                    # update status
                    polygonVerticalTut.status = FINISHED
                    polygonVerticalTut.setAutoDraw(False)
            
            # *gaborTut* updates
            
            # if gaborTut is starting this frame...
            if gaborTut.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                gaborTut.frameNStart = frameN  # exact frame index
                gaborTut.tStart = t  # local t and not account for scr refresh
                gaborTut.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(gaborTut, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'gaborTut.started')
                # update status
                gaborTut.status = STARTED
                gaborTut.setAutoDraw(True)
            
            # if gaborTut is active this frame...
            if gaborTut.status == STARTED:
                # update params
                gaborTut.setPos(positionT, log=False)
                gaborTut.setSize(sizeT, log=False)
                gaborTut.setOri(orientationT, log=False)
                gaborTut.setTex(textureT, log=False)
                gaborTut.setMask(maskT, log=False)
                gaborTut.setSF(spacial_freqT, log=False)
                gaborTut.setPhase(0.0, log=False)
            
            # if gaborTut is stopping this frame...
            if gaborTut.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > gaborTut.tStartRefresh + 10-frameTolerance:
                    # keep track of stop time/frame for later
                    gaborTut.tStop = t  # not accounting for scr refresh
                    gaborTut.tStopRefresh = tThisFlipGlobal  # on global time
                    gaborTut.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'gaborTut.stopped')
                    # update status
                    gaborTut.status = FINISHED
                    gaborTut.setAutoDraw(False)
            # Run 'Each Frame' code from codeMoveT
            pressed_keys = kbd.getKeys(keyList=['left', 'right'], waitRelease=False, clear=False)
            
            step = 0.5 
            
            if pressed_keys:
                active = [t.name for t in pressed_keys if t.duration is None]
                
                if 'left' in active:
                    pos_x -= step
                if 'right' in active:
                    pos_x += step
            
            gabor.pos = [pos_x, pos_y]
            
            if positionT[0] > 0 and pos_x < 0.0:
                pos_x = 0.0
                continueRoutine = False
            
            elif positionT[0] < 0 and pos_x > 0.0:
                pos_x = 0.0
                continueRoutine = False
            
            
            gabor.pos = [pos_x, pos_y]
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=Tutorial,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                Tutorial.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if Tutorial.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in Tutorial.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Tutorial" ---
        for thisComponent in Tutorial.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Tutorial
        Tutorial.tStop = globalClock.getTime(format='float')
        Tutorial.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Tutorial.stopped', Tutorial.tStop)
        soundTutGoCue.pause()  # ensure sound has stopped at end of Routine
        # Run 'End Routine' code from codeCorKeyTut
        corans = False
        
        if positionT[0] > 0 and gabor.pos[0] == 0.0:
            corans = True
        
        
        elif positionT[0] < 0 and gabor.pos[0] == 0.0:
            corans = True
        
        
        
        
        thisExp.addData('Correct answer', corans)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if Tutorial.maxDurationReached:
            routineTimer.addTime(-Tutorial.maxDuration)
        elif Tutorial.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-11.000000)
        
        # --- Prepare to start Routine "TutFeedback" ---
        # create an object to store info about Routine TutFeedback
        TutFeedback = data.Routine(
            name='TutFeedback',
            components=[textFBTut, soundFBTut],
        )
        TutFeedback.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from codeFcondTut
        if corans:
            scoreT += 1
            text_feedback = f"Correto!\n+1 point\nTotal: {scoreT}"
            win.color = "green"
            sound_feedback = 4000
            durationtextT = 0.5
            volume = 1
            duration = 0.25
        else:
            scoreT += 0
            text_feedback = f"Errado!\n+0 points\nTotal: {scoreT}"
            win.color = "red"
            sound_feedback = "sound_files/ibl_noise_burst.wav"
            durationtextT = 1
            volume = 1
            duration = 0.5
        
        win.flip()
        soundFBTut.setSound(sound_feedback , secs=duration, hamming=True)
        soundFBTut.setVolume(1.0, log=False)
        soundFBTut.seek(0)
        # store start times for TutFeedback
        TutFeedback.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        TutFeedback.tStart = globalClock.getTime(format='float')
        TutFeedback.status = STARTED
        thisExp.addData('TutFeedback.started', TutFeedback.tStart)
        TutFeedback.maxDuration = None
        # keep track of which components have finished
        TutFeedbackComponents = TutFeedback.components
        for thisComponent in TutFeedback.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "TutFeedback" ---
        thisExp.currentRoutine = TutFeedback
        TutFeedback.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisPreTest1TrialsTut, 'status') and thisPreTest1TrialsTut.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *textFBTut* updates
            
            # if textFBTut is starting this frame...
            if textFBTut.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                textFBTut.frameNStart = frameN  # exact frame index
                textFBTut.tStart = t  # local t and not account for scr refresh
                textFBTut.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(textFBTut, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textFBTut.started')
                # update status
                textFBTut.status = STARTED
                textFBTut.setAutoDraw(True)
            
            # if textFBTut is active this frame...
            if textFBTut.status == STARTED:
                # update params
                textFBTut.setText(text_feedback, log=False)
            
            # if textFBTut is stopping this frame...
            if textFBTut.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > textFBTut.tStartRefresh + durationtextT-frameTolerance:
                    # keep track of stop time/frame for later
                    textFBTut.tStop = t  # not accounting for scr refresh
                    textFBTut.tStopRefresh = tThisFlipGlobal  # on global time
                    textFBTut.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textFBTut.stopped')
                    # update status
                    textFBTut.status = FINISHED
                    textFBTut.setAutoDraw(False)
            
            # *soundFBTut* updates
            
            # if soundFBTut is starting this frame...
            if soundFBTut.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                soundFBTut.frameNStart = frameN  # exact frame index
                soundFBTut.tStart = t  # local t and not account for scr refresh
                soundFBTut.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('soundFBTut.started', tThisFlipGlobal)
                # update status
                soundFBTut.status = STARTED
                soundFBTut.play(when=win)  # sync with win flip
            
            # if soundFBTut is stopping this frame...
            if soundFBTut.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > soundFBTut.tStartRefresh + duration-frameTolerance or soundFBTut.isFinished:
                    # keep track of stop time/frame for later
                    soundFBTut.tStop = t  # not accounting for scr refresh
                    soundFBTut.tStopRefresh = tThisFlipGlobal  # on global time
                    soundFBTut.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'soundFBTut.stopped')
                    # update status
                    soundFBTut.status = FINISHED
                    soundFBTut.stop()
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=TutFeedback,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                TutFeedback.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if TutFeedback.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in TutFeedback.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "TutFeedback" ---
        for thisComponent in TutFeedback.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for TutFeedback
        TutFeedback.tStop = globalClock.getTime(format='float')
        TutFeedback.tStopRefresh = tThisFlipGlobal
        thisExp.addData('TutFeedback.stopped', TutFeedback.tStop)
        # Run 'End Routine' code from codeFcondTut
        win.color="gray"
        soundFBTut.pause()  # ensure sound has stopped at end of Routine
        # the Routine "TutFeedback" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisPreTest1TrialsTut as finished
        if hasattr(thisPreTest1TrialsTut, 'status'):
            thisPreTest1TrialsTut.status = FINISHED
        # if awaiting a pause, pause now
        if PreTest1TrialsTut.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            PreTest1TrialsTut.status = STARTED
        thisExp.nextEntry()
        
    # completed 3 repeats of 'PreTest1TrialsTut'
    PreTest1TrialsTut.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "Blank4000" ---
    # create an object to store info about Routine Blank4000
    Blank4000 = data.Routine(
        name='Blank4000',
        components=[textBlank],
    )
    Blank4000.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for Blank4000
    Blank4000.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Blank4000.tStart = globalClock.getTime(format='float')
    Blank4000.status = STARTED
    thisExp.addData('Blank4000.started', Blank4000.tStart)
    Blank4000.maxDuration = None
    # keep track of which components have finished
    Blank4000Components = Blank4000.components
    for thisComponent in Blank4000.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Blank4000" ---
    thisExp.currentRoutine = Blank4000
    Blank4000.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 4.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *textBlank* updates
        
        # if textBlank is starting this frame...
        if textBlank.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            textBlank.frameNStart = frameN  # exact frame index
            textBlank.tStart = t  # local t and not account for scr refresh
            textBlank.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(textBlank, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'textBlank.started')
            # update status
            textBlank.status = STARTED
            textBlank.setAutoDraw(True)
        
        # if textBlank is active this frame...
        if textBlank.status == STARTED:
            # update params
            pass
        
        # if textBlank is stopping this frame...
        if textBlank.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > textBlank.tStartRefresh + 4-frameTolerance:
                # keep track of stop time/frame for later
                textBlank.tStop = t  # not accounting for scr refresh
                textBlank.tStopRefresh = tThisFlipGlobal  # on global time
                textBlank.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textBlank.stopped')
                # update status
                textBlank.status = FINISHED
                textBlank.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=Blank4000,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            Blank4000.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if Blank4000.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in Blank4000.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Blank4000" ---
    for thisComponent in Blank4000.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Blank4000
    Blank4000.tStop = globalClock.getTime(format='float')
    Blank4000.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Blank4000.stopped', Blank4000.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if Blank4000.maxDurationReached:
        routineTimer.addTime(-Blank4000.maxDuration)
    elif Blank4000.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-4.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "InstructionsFullTask" ---
    # create an object to store info about Routine InstructionsFullTask
    InstructionsFullTask = data.Routine(
        name='InstructionsFullTask',
        components=[textFullTask, keyRespFullTask],
    )
    InstructionsFullTask.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for keyRespFullTask
    keyRespFullTask.keys = []
    keyRespFullTask.rt = []
    _keyRespFullTask_allKeys = []
    # store start times for InstructionsFullTask
    InstructionsFullTask.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    InstructionsFullTask.tStart = globalClock.getTime(format='float')
    InstructionsFullTask.status = STARTED
    thisExp.addData('InstructionsFullTask.started', InstructionsFullTask.tStart)
    InstructionsFullTask.maxDuration = None
    # keep track of which components have finished
    InstructionsFullTaskComponents = InstructionsFullTask.components
    for thisComponent in InstructionsFullTask.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "InstructionsFullTask" ---
    thisExp.currentRoutine = InstructionsFullTask
    InstructionsFullTask.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *textFullTask* updates
        
        # if textFullTask is starting this frame...
        if textFullTask.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            textFullTask.frameNStart = frameN  # exact frame index
            textFullTask.tStart = t  # local t and not account for scr refresh
            textFullTask.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(textFullTask, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'textFullTask.started')
            # update status
            textFullTask.status = STARTED
            textFullTask.setAutoDraw(True)
        
        # if textFullTask is active this frame...
        if textFullTask.status == STARTED:
            # update params
            pass
        
        # *keyRespFullTask* updates
        waitOnFlip = False
        
        # if keyRespFullTask is starting this frame...
        if keyRespFullTask.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            keyRespFullTask.frameNStart = frameN  # exact frame index
            keyRespFullTask.tStart = t  # local t and not account for scr refresh
            keyRespFullTask.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(keyRespFullTask, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'keyRespFullTask.started')
            # update status
            keyRespFullTask.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(keyRespFullTask.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(keyRespFullTask.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if keyRespFullTask.status == STARTED and not waitOnFlip:
            theseKeys = keyRespFullTask.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _keyRespFullTask_allKeys.extend(theseKeys)
            if len(_keyRespFullTask_allKeys):
                keyRespFullTask.keys = _keyRespFullTask_allKeys[-1].name  # just the last key pressed
                keyRespFullTask.rt = _keyRespFullTask_allKeys[-1].rt
                keyRespFullTask.duration = _keyRespFullTask_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=InstructionsFullTask,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            InstructionsFullTask.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if InstructionsFullTask.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in InstructionsFullTask.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "InstructionsFullTask" ---
    for thisComponent in InstructionsFullTask.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for InstructionsFullTask
    InstructionsFullTask.tStop = globalClock.getTime(format='float')
    InstructionsFullTask.tStopRefresh = tThisFlipGlobal
    thisExp.addData('InstructionsFullTask.stopped', InstructionsFullTask.tStop)
    # check responses
    if keyRespFullTask.keys in ['', [], None]:  # No response was made
        keyRespFullTask.keys = None
    thisExp.addData('keyRespFullTask.keys',keyRespFullTask.keys)
    if keyRespFullTask.keys != None:  # we had a response
        thisExp.addData('keyRespFullTask.rt', keyRespFullTask.rt)
        thisExp.addData('keyRespFullTask.duration', keyRespFullTask.duration)
    thisExp.nextEntry()
    # the Routine "InstructionsFullTask" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    Pretest1Block = data.TrialHandler2(
        name='Pretest1Block',
        nReps=15, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(Pretest1Block)  # add the loop to the experiment
    thisPretest1Block = Pretest1Block.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPretest1Block.rgb)
    if thisPretest1Block != None:
        for paramName in thisPretest1Block:
            globals()[paramName] = thisPretest1Block[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisPretest1Block in Pretest1Block:
        Pretest1Block.status = STARTED
        if hasattr(thisPretest1Block, 'status'):
            thisPretest1Block.status = STARTED
        currentLoop = Pretest1Block
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisPretest1Block.rgb)
        if thisPretest1Block != None:
            for paramName in thisPretest1Block:
                globals()[paramName] = thisPretest1Block[paramName]
        
        # set up handler to look after randomisation of conditions etc
        pretest1trialsTraining = data.TrialHandler2(
            name='pretest1trialsTraining',
            nReps=2, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions('Conditions_parameters.xlsx'), 
            seed=None, 
            isTrials=True, 
        )
        thisExp.addLoop(pretest1trialsTraining)  # add the loop to the experiment
        thisPretest1trialsTraining = pretest1trialsTraining.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisPretest1trialsTraining.rgb)
        if thisPretest1trialsTraining != None:
            for paramName in thisPretest1trialsTraining:
                globals()[paramName] = thisPretest1trialsTraining[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisPretest1trialsTraining in pretest1trialsTraining:
            pretest1trialsTraining.status = STARTED
            if hasattr(thisPretest1trialsTraining, 'status'):
                thisPretest1trialsTraining.status = STARTED
            currentLoop = pretest1trialsTraining
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisPretest1trialsTraining.rgb)
            if thisPretest1trialsTraining != None:
                for paramName in thisPretest1trialsTraining:
                    globals()[paramName] = thisPretest1trialsTraining[paramName]
            
            # --- Prepare to start Routine "FullTask" ---
            # create an object to store info about Routine FullTask
            FullTask = data.Routine(
                name='FullTask',
                components=[soundGoCue, polygonVertical, polygonHorizontal, gabor],
            )
            FullTask.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            soundGoCue.setSound('5000', secs=0.1, hamming=True)
            soundGoCue.setVolume(1.0, log=False)
            soundGoCue.seek(0)
            gabor.setContrast(grating)
            # Run 'Begin Routine' code from codePhase
            # Generate random phase
            current_phase = np.random.uniform(0, 2 * np.pi)
            
            gabor.phase = current_phase
            
            # Creates a gabor phase column on the csv
            thisExp.addData('gabor_phase', current_phase)
            # Run 'Begin Routine' code from codeMove
            try:
            
                if hasattr(position[0], '__iter__') or isinstance(position[0], (list, tuple)):
                    pos_x = float(position[0][0])
                    pos_y = float(position[0][1])
                else:
                    pos_x = float(position[0])
                    pos_y = float(position[1])
            except Exception:
                pos_x = 0.0
                pos_y = 0.0
            
            
            kbd.clearEvents()
            
            # store start times for FullTask
            FullTask.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            FullTask.tStart = globalClock.getTime(format='float')
            FullTask.status = STARTED
            thisExp.addData('FullTask.started', FullTask.tStart)
            FullTask.maxDuration = None
            # keep track of which components have finished
            FullTaskComponents = FullTask.components
            for thisComponent in FullTask.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "FullTask" ---
            thisExp.currentRoutine = FullTask
            FullTask.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 4.0:
                # if trial has changed, end Routine now
                if hasattr(thisPretest1trialsTraining, 'status') and thisPretest1trialsTraining.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *soundGoCue* updates
                
                # if soundGoCue is starting this frame...
                if soundGoCue.status == NOT_STARTED and tThisFlip >= 0.1-frameTolerance:
                    # keep track of start time/frame for later
                    soundGoCue.frameNStart = frameN  # exact frame index
                    soundGoCue.tStart = t  # local t and not account for scr refresh
                    soundGoCue.tStartRefresh = tThisFlipGlobal  # on global time
                    # add timestamp to datafile
                    thisExp.addData('soundGoCue.started', tThisFlipGlobal)
                    # update status
                    soundGoCue.status = STARTED
                    soundGoCue.play(when=win)  # sync with win flip
                
                # if soundGoCue is stopping this frame...
                if soundGoCue.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > soundGoCue.tStartRefresh + 0.1-frameTolerance or soundGoCue.isFinished:
                        # keep track of stop time/frame for later
                        soundGoCue.tStop = t  # not accounting for scr refresh
                        soundGoCue.tStopRefresh = tThisFlipGlobal  # on global time
                        soundGoCue.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'soundGoCue.stopped')
                        # update status
                        soundGoCue.status = FINISHED
                        soundGoCue.stop()
                
                # *polygonVertical* updates
                
                # if polygonVertical is starting this frame...
                if polygonVertical.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    polygonVertical.frameNStart = frameN  # exact frame index
                    polygonVertical.tStart = t  # local t and not account for scr refresh
                    polygonVertical.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(polygonVertical, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    polygonVertical.status = STARTED
                    polygonVertical.setAutoDraw(True)
                
                # if polygonVertical is active this frame...
                if polygonVertical.status == STARTED:
                    # update params
                    polygonVertical.setPos(fixation_pos, log=False)
                    polygonVertical.setSize(fixation_size, log=False)
                    polygonVertical.setOri(90.0, log=False)
                    polygonVertical.setLineWidth(fixation_width, log=False)
                
                # if polygonVertical is stopping this frame...
                if polygonVertical.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > polygonVertical.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        polygonVertical.tStop = t  # not accounting for scr refresh
                        polygonVertical.tStopRefresh = tThisFlipGlobal  # on global time
                        polygonVertical.frameNStop = frameN  # exact frame index
                        # update status
                        polygonVertical.status = FINISHED
                        polygonVertical.setAutoDraw(False)
                
                # *polygonHorizontal* updates
                
                # if polygonHorizontal is starting this frame...
                if polygonHorizontal.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    polygonHorizontal.frameNStart = frameN  # exact frame index
                    polygonHorizontal.tStart = t  # local t and not account for scr refresh
                    polygonHorizontal.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(polygonHorizontal, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    polygonHorizontal.status = STARTED
                    polygonHorizontal.setAutoDraw(True)
                
                # if polygonHorizontal is active this frame...
                if polygonHorizontal.status == STARTED:
                    # update params
                    polygonHorizontal.setPos(fixation_pos, log=False)
                    polygonHorizontal.setSize(fixation_size, log=False)
                    polygonHorizontal.setOri(180.0, log=False)
                    polygonHorizontal.setLineWidth(fixation_width, log=False)
                
                # if polygonHorizontal is stopping this frame...
                if polygonHorizontal.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > polygonHorizontal.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        polygonHorizontal.tStop = t  # not accounting for scr refresh
                        polygonHorizontal.tStopRefresh = tThisFlipGlobal  # on global time
                        polygonHorizontal.frameNStop = frameN  # exact frame index
                        # update status
                        polygonHorizontal.status = FINISHED
                        polygonHorizontal.setAutoDraw(False)
                
                # *gabor* updates
                
                # if gabor is starting this frame...
                if gabor.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                    # keep track of start time/frame for later
                    gabor.frameNStart = frameN  # exact frame index
                    gabor.tStart = t  # local t and not account for scr refresh
                    gabor.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(gabor, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'gabor.started')
                    # update status
                    gabor.status = STARTED
                    gabor.setAutoDraw(True)
                
                # if gabor is active this frame...
                if gabor.status == STARTED:
                    # update params
                    gabor.setPos(position, log=False)
                    gabor.setSize(size, log=False)
                    gabor.setOri(orientation, log=False)
                    gabor.setTex(texture, log=False)
                    gabor.setMask(mask, log=False)
                    gabor.setSF(spacial_freq, log=False)
                    gabor.setPhase(0.0, log=False)
                
                # if gabor is stopping this frame...
                if gabor.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > gabor.tStartRefresh + 3.00-frameTolerance:
                        # keep track of stop time/frame for later
                        gabor.tStop = t  # not accounting for scr refresh
                        gabor.tStopRefresh = tThisFlipGlobal  # on global time
                        gabor.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'gabor.stopped')
                        # update status
                        gabor.status = FINISHED
                        gabor.setAutoDraw(False)
                # Run 'Each Frame' code from codeMove
                pressed_keys = kbd.getKeys(keyList=['left', 'right'], waitRelease=False, clear=False)
                
                step = 0.5 
                
                if pressed_keys:
                    active = [t.name for t in pressed_keys if t.duration is None]
                    
                    if 'left' in active:
                        pos_x -= step
                    if 'right' in active:
                        pos_x += step
                
                gabor.pos = [pos_x, pos_y]
                
                if position[0] > 0 and pos_x < 0.0:
                    pos_x = 0.0
                    continueRoutine = False
                
                elif position[0] < 0 and pos_x > 0.0:
                    pos_x = 0.0
                    continueRoutine = False
                
                
                gabor.pos = [pos_x, pos_y]
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=FullTask,
                    )
                    # skip the frame we paused on
                    continue
                
                # has a Component requested the Routine to end?
                if not continueRoutine:
                    FullTask.forceEnded = routineForceEnded = True
                # has the Routine been forcibly ended?
                if FullTask.forceEnded or routineForceEnded:
                    break
                # has every Component finished?
                continueRoutine = False
                for thisComponent in FullTask.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "FullTask" ---
            for thisComponent in FullTask.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for FullTask
            FullTask.tStop = globalClock.getTime(format='float')
            FullTask.tStopRefresh = tThisFlipGlobal
            thisExp.addData('FullTask.stopped', FullTask.tStop)
            soundGoCue.pause()  # ensure sound has stopped at end of Routine
            # Run 'End Routine' code from codeCorKey
            corans = False
            
            if position[0] > 0 and gabor.pos[0] == 0.0:
                corans = True
            
            
            elif position[0] < 0 and gabor.pos[0] == 0.0:
                corans = True
            
            
            
            
            thisExp.addData('Correct answer', corans)
            
            total_task_time = total_task_timer.getTime()
            print(f"TOTAL TASK TIME: {total_task_time:.2f} SECONDS\n")
            
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if FullTask.maxDurationReached:
                routineTimer.addTime(-FullTask.maxDuration)
            elif FullTask.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-4.000000)
            
            # --- Prepare to start Routine "FullTaskFeedback" ---
            # create an object to store info about Routine FullTaskFeedback
            FullTaskFeedback = data.Routine(
                name='FullTaskFeedback',
                components=[textFeedback, soundFeedback],
            )
            FullTaskFeedback.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from codeFconditions
            if corans:
                score += 1
                text_feedback = f"Correto!\n+1 point\nTotal: {score}"
                win.color = "green"
                sound_feedback = 4000
                durationtext = 0.5
                volume = 1
                duration = 0.25
            else:
                score += 0
                text_feedback = f"Errado!\n+0 points\nTotal: {score}"
                win.color = "red"
                sound_feedback = "sound_files/ibl_noise_burst.wav"
                durationtext = 1.5
                volume = 1
                duration = 0.5
            
                
            win.flip()
            
            soundFeedback.setSound(sound_feedback , secs=duration, hamming=True)
            soundFeedback.setVolume(1.0, log=False)
            soundFeedback.seek(0)
            # store start times for FullTaskFeedback
            FullTaskFeedback.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            FullTaskFeedback.tStart = globalClock.getTime(format='float')
            FullTaskFeedback.status = STARTED
            thisExp.addData('FullTaskFeedback.started', FullTaskFeedback.tStart)
            FullTaskFeedback.maxDuration = None
            # keep track of which components have finished
            FullTaskFeedbackComponents = FullTaskFeedback.components
            for thisComponent in FullTaskFeedback.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "FullTaskFeedback" ---
            thisExp.currentRoutine = FullTaskFeedback
            FullTaskFeedback.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisPretest1trialsTraining, 'status') and thisPretest1trialsTraining.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *textFeedback* updates
                
                # if textFeedback is starting this frame...
                if textFeedback.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    textFeedback.frameNStart = frameN  # exact frame index
                    textFeedback.tStart = t  # local t and not account for scr refresh
                    textFeedback.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(textFeedback, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textFeedback.started')
                    # update status
                    textFeedback.status = STARTED
                    textFeedback.setAutoDraw(True)
                
                # if textFeedback is active this frame...
                if textFeedback.status == STARTED:
                    # update params
                    textFeedback.setText(text_feedback, log=False)
                
                # if textFeedback is stopping this frame...
                if textFeedback.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > textFeedback.tStartRefresh + durationtext-frameTolerance:
                        # keep track of stop time/frame for later
                        textFeedback.tStop = t  # not accounting for scr refresh
                        textFeedback.tStopRefresh = tThisFlipGlobal  # on global time
                        textFeedback.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'textFeedback.stopped')
                        # update status
                        textFeedback.status = FINISHED
                        textFeedback.setAutoDraw(False)
                
                # *soundFeedback* updates
                
                # if soundFeedback is starting this frame...
                if soundFeedback.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    soundFeedback.frameNStart = frameN  # exact frame index
                    soundFeedback.tStart = t  # local t and not account for scr refresh
                    soundFeedback.tStartRefresh = tThisFlipGlobal  # on global time
                    # add timestamp to datafile
                    thisExp.addData('soundFeedback.started', tThisFlipGlobal)
                    # update status
                    soundFeedback.status = STARTED
                    soundFeedback.play(when=win)  # sync with win flip
                
                # if soundFeedback is stopping this frame...
                if soundFeedback.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > soundFeedback.tStartRefresh + duration-frameTolerance or soundFeedback.isFinished:
                        # keep track of stop time/frame for later
                        soundFeedback.tStop = t  # not accounting for scr refresh
                        soundFeedback.tStopRefresh = tThisFlipGlobal  # on global time
                        soundFeedback.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'soundFeedback.stopped')
                        # update status
                        soundFeedback.status = FINISHED
                        soundFeedback.stop()
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=FullTaskFeedback,
                    )
                    # skip the frame we paused on
                    continue
                
                # has a Component requested the Routine to end?
                if not continueRoutine:
                    FullTaskFeedback.forceEnded = routineForceEnded = True
                # has the Routine been forcibly ended?
                if FullTaskFeedback.forceEnded or routineForceEnded:
                    break
                # has every Component finished?
                continueRoutine = False
                for thisComponent in FullTaskFeedback.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "FullTaskFeedback" ---
            for thisComponent in FullTaskFeedback.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for FullTaskFeedback
            FullTaskFeedback.tStop = globalClock.getTime(format='float')
            FullTaskFeedback.tStopRefresh = tThisFlipGlobal
            thisExp.addData('FullTaskFeedback.stopped', FullTaskFeedback.tStop)
            # Run 'End Routine' code from codeFconditions
            win.color="gray"
            soundFeedback.pause()  # ensure sound has stopped at end of Routine
            # the Routine "FullTaskFeedback" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            # mark thisPretest1trialsTraining as finished
            if hasattr(thisPretest1trialsTraining, 'status'):
                thisPretest1trialsTraining.status = FINISHED
            # if awaiting a pause, pause now
            if pretest1trialsTraining.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                pretest1trialsTraining.status = STARTED
            thisExp.nextEntry()
            
        # completed 2 repeats of 'pretest1trialsTraining'
        pretest1trialsTraining.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        # --- Prepare to start Routine "Pause" ---
        # create an object to store info about Routine Pause
        Pause = data.Routine(
            name='Pause',
            components=[textPause, Returntotasktext, key_resp],
        )
        Pause.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from codeLevels
        # Lista de níveis atualizada sem cores cinzentas
        levels_config = [
            {"min_score": 0,   "name": "Recruta",      "color": "#FFFFFF"},  # Branco (Contraste total com o fundo cinzento)
            {"min_score": 50,  "name": "Iniciante",    "color": "#CD7F32"},  # Bronze / Castanho Quente
            {"min_score": 100, "name": "Explorador",   "color": "#4CAF50"},  # Verde
            {"min_score": 150, "name": "Praticante",   "color": "#00BCD4"},  # Ciano / Azul Claro
            {"min_score": 200, "name": "Especialista", "color": "#2196F3"},  # Azul
            {"min_score": 250, "name": "Perito",       "color": "#9C27B0"},  # Roxo
            {"min_score": 300, "name": "Mestre",       "color": "#E91E63"},  # Rosa Forte
            {"min_score": 350, "name": "Guru",         "color": "#FF9800"},  # Laranja
            {"min_score": 400, "name": "Lenda",        "color": "#FFD700"}   # Dourado
        ]
        
        current_level = levels_config[0]
        next_level = None
        
        for level in levels_config:
            if score >= level["min_score"]:
                current_level = level
            else:
                next_level = level
                break
        
        level_name = current_level["name"]
        level_color = current_level["color"]  
        
        
        text_pause = f"Pausa\nPossui neste momento {score} pontos\n"
        
        if score >= 400:
            text_pause += f"Nível {level_name}\nNível Máximo Atingido!"
        else:
            pontos_em_falta = next_level["min_score"] - score
            text_pause += f"Nível {level_name}\nFaltam {pontos_em_falta} pontos para o próximo nível"
        
        # create starting attributes for key_resp
        key_resp.keys = []
        key_resp.rt = []
        _key_resp_allKeys = []
        # store start times for Pause
        Pause.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Pause.tStart = globalClock.getTime(format='float')
        Pause.status = STARTED
        thisExp.addData('Pause.started', Pause.tStart)
        Pause.maxDuration = None
        # keep track of which components have finished
        PauseComponents = Pause.components
        for thisComponent in Pause.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Pause" ---
        thisExp.currentRoutine = Pause
        Pause.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisPretest1Block, 'status') and thisPretest1Block.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *textPause* updates
            
            # if textPause is starting this frame...
            if textPause.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                textPause.frameNStart = frameN  # exact frame index
                textPause.tStart = t  # local t and not account for scr refresh
                textPause.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(textPause, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textPause.started')
                # update status
                textPause.status = STARTED
                textPause.setAutoDraw(True)
            
            # if textPause is active this frame...
            if textPause.status == STARTED:
                # update params
                textPause.setText(text_pause, log=False)
            
            # if textPause is stopping this frame...
            if textPause.status == STARTED:
                # is it time to stop? (based on local clock)
                if tThisFlip > 15-frameTolerance:
                    # keep track of stop time/frame for later
                    textPause.tStop = t  # not accounting for scr refresh
                    textPause.tStopRefresh = tThisFlipGlobal  # on global time
                    textPause.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textPause.stopped')
                    # update status
                    textPause.status = FINISHED
                    textPause.setAutoDraw(False)
            
            # *Returntotasktext* updates
            
            # if Returntotasktext is starting this frame...
            if Returntotasktext.status == NOT_STARTED and tThisFlip >= 15-frameTolerance:
                # keep track of start time/frame for later
                Returntotasktext.frameNStart = frameN  # exact frame index
                Returntotasktext.tStart = t  # local t and not account for scr refresh
                Returntotasktext.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Returntotasktext, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Returntotasktext.started')
                # update status
                Returntotasktext.status = STARTED
                Returntotasktext.setAutoDraw(True)
            
            # if Returntotasktext is active this frame...
            if Returntotasktext.status == STARTED:
                # update params
                pass
            
            # if Returntotasktext is stopping this frame...
            if Returntotasktext.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Returntotasktext.tStartRefresh + 20-frameTolerance:
                    # keep track of stop time/frame for later
                    Returntotasktext.tStop = t  # not accounting for scr refresh
                    Returntotasktext.tStopRefresh = tThisFlipGlobal  # on global time
                    Returntotasktext.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Returntotasktext.stopped')
                    # update status
                    Returntotasktext.status = FINISHED
                    Returntotasktext.setAutoDraw(False)
            
            # *key_resp* updates
            waitOnFlip = False
            
            # if key_resp is starting this frame...
            if key_resp.status == NOT_STARTED and tThisFlip >= 15-frameTolerance:
                # keep track of start time/frame for later
                key_resp.frameNStart = frameN  # exact frame index
                key_resp.tStart = t  # local t and not account for scr refresh
                key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp.started')
                # update status
                key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp.status == STARTED and not waitOnFlip:
                theseKeys = key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_allKeys.extend(theseKeys)
                if len(_key_resp_allKeys):
                    key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                    key_resp.rt = _key_resp_allKeys[-1].rt
                    key_resp.duration = _key_resp_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=Pause,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                Pause.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if Pause.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in Pause.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Pause" ---
        for thisComponent in Pause.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Pause
        Pause.tStop = globalClock.getTime(format='float')
        Pause.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Pause.stopped', Pause.tStop)
        # check responses
        if key_resp.keys in ['', [], None]:  # No response was made
            key_resp.keys = None
        Pretest1Block.addData('key_resp.keys',key_resp.keys)
        if key_resp.keys != None:  # we had a response
            Pretest1Block.addData('key_resp.rt', key_resp.rt)
            Pretest1Block.addData('key_resp.duration', key_resp.duration)
        # the Routine "Pause" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisPretest1Block as finished
        if hasattr(thisPretest1Block, 'status'):
            thisPretest1Block.status = FINISHED
        # if awaiting a pause, pause now
        if Pretest1Block.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            Pretest1Block.status = STARTED
        thisExp.nextEntry()
        
    # completed 15 repeats of 'Pretest1Block'
    Pretest1Block.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "Blank4000" ---
    # create an object to store info about Routine Blank4000
    Blank4000 = data.Routine(
        name='Blank4000',
        components=[textBlank],
    )
    Blank4000.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for Blank4000
    Blank4000.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Blank4000.tStart = globalClock.getTime(format='float')
    Blank4000.status = STARTED
    thisExp.addData('Blank4000.started', Blank4000.tStart)
    Blank4000.maxDuration = None
    # keep track of which components have finished
    Blank4000Components = Blank4000.components
    for thisComponent in Blank4000.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Blank4000" ---
    thisExp.currentRoutine = Blank4000
    Blank4000.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 4.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *textBlank* updates
        
        # if textBlank is starting this frame...
        if textBlank.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            textBlank.frameNStart = frameN  # exact frame index
            textBlank.tStart = t  # local t and not account for scr refresh
            textBlank.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(textBlank, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'textBlank.started')
            # update status
            textBlank.status = STARTED
            textBlank.setAutoDraw(True)
        
        # if textBlank is active this frame...
        if textBlank.status == STARTED:
            # update params
            pass
        
        # if textBlank is stopping this frame...
        if textBlank.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > textBlank.tStartRefresh + 4-frameTolerance:
                # keep track of stop time/frame for later
                textBlank.tStop = t  # not accounting for scr refresh
                textBlank.tStopRefresh = tThisFlipGlobal  # on global time
                textBlank.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textBlank.stopped')
                # update status
                textBlank.status = FINISHED
                textBlank.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=Blank4000,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            Blank4000.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if Blank4000.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in Blank4000.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Blank4000" ---
    for thisComponent in Blank4000.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Blank4000
    Blank4000.tStop = globalClock.getTime(format='float')
    Blank4000.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Blank4000.stopped', Blank4000.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if Blank4000.maxDurationReached:
        routineTimer.addTime(-Blank4000.maxDuration)
    elif Blank4000.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-4.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "GoodbyeScreen" ---
    # create an object to store info about Routine GoodbyeScreen
    GoodbyeScreen = data.Routine(
        name='GoodbyeScreen',
        components=[textEnd, keyLeaderBoard],
    )
    GoodbyeScreen.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for keyLeaderBoard
    keyLeaderBoard.keys = []
    keyLeaderBoard.rt = []
    _keyLeaderBoard_allKeys = []
    # store start times for GoodbyeScreen
    GoodbyeScreen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    GoodbyeScreen.tStart = globalClock.getTime(format='float')
    GoodbyeScreen.status = STARTED
    thisExp.addData('GoodbyeScreen.started', GoodbyeScreen.tStart)
    GoodbyeScreen.maxDuration = None
    # keep track of which components have finished
    GoodbyeScreenComponents = GoodbyeScreen.components
    for thisComponent in GoodbyeScreen.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "GoodbyeScreen" ---
    thisExp.currentRoutine = GoodbyeScreen
    GoodbyeScreen.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *textEnd* updates
        
        # if textEnd is starting this frame...
        if textEnd.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            textEnd.frameNStart = frameN  # exact frame index
            textEnd.tStart = t  # local t and not account for scr refresh
            textEnd.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(textEnd, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'textEnd.started')
            # update status
            textEnd.status = STARTED
            textEnd.setAutoDraw(True)
        
        # if textEnd is active this frame...
        if textEnd.status == STARTED:
            # update params
            pass
        
        # *keyLeaderBoard* updates
        waitOnFlip = False
        
        # if keyLeaderBoard is starting this frame...
        if keyLeaderBoard.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            keyLeaderBoard.frameNStart = frameN  # exact frame index
            keyLeaderBoard.tStart = t  # local t and not account for scr refresh
            keyLeaderBoard.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(keyLeaderBoard, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'keyLeaderBoard.started')
            # update status
            keyLeaderBoard.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(keyLeaderBoard.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(keyLeaderBoard.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if keyLeaderBoard.status == STARTED and not waitOnFlip:
            theseKeys = keyLeaderBoard.getKeys(keyList=['y','n'], ignoreKeys=["escape"], waitRelease=False)
            _keyLeaderBoard_allKeys.extend(theseKeys)
            if len(_keyLeaderBoard_allKeys):
                keyLeaderBoard.keys = _keyLeaderBoard_allKeys[0].name  # just the first key pressed
                keyLeaderBoard.rt = _keyLeaderBoard_allKeys[0].rt
                keyLeaderBoard.duration = _keyLeaderBoard_allKeys[0].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=GoodbyeScreen,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            GoodbyeScreen.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if GoodbyeScreen.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in GoodbyeScreen.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "GoodbyeScreen" ---
    for thisComponent in GoodbyeScreen.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for GoodbyeScreen
    GoodbyeScreen.tStop = globalClock.getTime(format='float')
    GoodbyeScreen.tStopRefresh = tThisFlipGlobal
    thisExp.addData('GoodbyeScreen.stopped', GoodbyeScreen.tStop)
    # Run 'End Routine' code from codepartID
    selected_key = None
    
    
    if keyLeaderBoard.keys and len(keyLeaderBoard.keys) > 0:
        if isinstance(keyLeaderBoard.keys, list):
            selected_key = keyLeaderBoard.keys[0]
        else:
            selected_key = keyLeaderBoard.keys
    
    
    end_msg = "Obrigado pela sua participação!"
    
    if selected_key == 'y':
        end_msg += f"\nEste é o seu ID de participante: {expInfo['participant']}.\n Guarde este número para verificar onde ficou colocado"
    elif selected_key == 'n':
        end_msg += f"\nO seu ID não aparecerá no leaderboard."
    
    # check responses
    if keyLeaderBoard.keys in ['', [], None]:  # No response was made
        keyLeaderBoard.keys = None
    thisExp.addData('keyLeaderBoard.keys',keyLeaderBoard.keys)
    if keyLeaderBoard.keys != None:  # we had a response
        thisExp.addData('keyLeaderBoard.rt', keyLeaderBoard.rt)
        thisExp.addData('keyLeaderBoard.duration', keyLeaderBoard.duration)
    thisExp.nextEntry()
    # the Routine "GoodbyeScreen" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "IDScreen" ---
    # create an object to store info about Routine IDScreen
    IDScreen = data.Routine(
        name='IDScreen',
        components=[textGoodbye],
    )
    IDScreen.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for IDScreen
    IDScreen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    IDScreen.tStart = globalClock.getTime(format='float')
    IDScreen.status = STARTED
    thisExp.addData('IDScreen.started', IDScreen.tStart)
    IDScreen.maxDuration = None
    # keep track of which components have finished
    IDScreenComponents = IDScreen.components
    for thisComponent in IDScreen.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "IDScreen" ---
    thisExp.currentRoutine = IDScreen
    IDScreen.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 3.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *textGoodbye* updates
        
        # if textGoodbye is starting this frame...
        if textGoodbye.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            textGoodbye.frameNStart = frameN  # exact frame index
            textGoodbye.tStart = t  # local t and not account for scr refresh
            textGoodbye.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(textGoodbye, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'textGoodbye.started')
            # update status
            textGoodbye.status = STARTED
            textGoodbye.setAutoDraw(True)
        
        # if textGoodbye is active this frame...
        if textGoodbye.status == STARTED:
            # update params
            textGoodbye.setText(end_msg
            
            , log=False)
        
        # if textGoodbye is stopping this frame...
        if textGoodbye.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > textGoodbye.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                textGoodbye.tStop = t  # not accounting for scr refresh
                textGoodbye.tStopRefresh = tThisFlipGlobal  # on global time
                textGoodbye.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textGoodbye.stopped')
                # update status
                textGoodbye.status = FINISHED
                textGoodbye.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=IDScreen,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            IDScreen.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if IDScreen.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in IDScreen.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "IDScreen" ---
    for thisComponent in IDScreen.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for IDScreen
    IDScreen.tStop = globalClock.getTime(format='float')
    IDScreen.tStopRefresh = tThisFlipGlobal
    thisExp.addData('IDScreen.stopped', IDScreen.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if IDScreen.maxDurationReached:
        routineTimer.addTime(-IDScreen.maxDuration)
    elif IDScreen.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-3.000000)
    thisExp.nextEntry()
    # Run 'End Experiment' code from GlobalCam
    rec.stop()
    print("Saved Capture.")
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    # stop any playback components
    if thisExp.currentRoutine is not None:
        for comp in thisExp.currentRoutine.getPlaybackComponents():
            comp.stop()
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # run any 'at exit' functions
    for fcn in runAtExit:
        fcn()
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
