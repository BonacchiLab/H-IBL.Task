#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2026.1.2),
    on julho 23, 2026, at 15:46
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

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2026.1.2'
expName = 'TrainingTask_Test'  # from the Builder filename that created this script
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
        originPath='C:\\Users\\Asus\\Documents\\recursos\\H-IBL.Task\\TrainingTask_Test_lastrun.py',
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
    textExplanation = visual.TextStim(win=win, name='textExplanation',
        text='Irá iniciar a tarefa. A mesma consiste na apresentação de um contraste à direita/esquerda. Indique em cada trial o lado do estímulo utilizando a tecla -S- para o lado esquerdo e  -L- para o lado direito. \n\nPedimos que mantenha o olhar no centro do ecrã durante a experiência\n\nPressione a SPACEBAR para começar',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "TrainingTrials" ---
    polygonVertical = visual.Rect(
        win=win, name='polygonVertical',units='deg', 
        width=[1.0, 1.0][0], height=[1.0, 1.0][1],
        ori=1.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=(0.0000, 0.0000, 0.0000), fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    polygonHorizontal = visual.Rect(
        win=win, name='polygonHorizontal',units='deg', 
        width=[1.0, 1.0][0], height=[1.0, 1.0][1],
        ori=1.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=(0.0000, 0.0000, 0.0000), fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    gabor = visual.GratingStim(
        win=win, name='gabor',units='deg', 
        tex=None, mask='sin', anchor='center',
        ori=1.0, pos=[0,0], draggable=False, size=1.0, sf=1.0, phase=1.0,
        color=[1,1,1], colorSpace='rgb',
        opacity=None, contrast=1.0, blendmode='avg',
        texRes=256.0, interpolate=True, depth=-2.0)
    keySide = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "TrialFeedback" ---
    # Run 'Begin Experiment' code from codeFconditions
    score = 0
    textFeedback = visual.TextStim(win=win, name='textFeedback',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    # set audio backend
    sound.Sound.backend = 'ptb'
    soundFeedback = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker=None,    name='soundFeedback'
    )
    soundFeedback.setVolume(1.0)
    
    # --- Initialize components for Routine "Blank4000" ---
    textBlank = visual.TextStim(win=win, name='textBlank',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "GoodbyeScreen" ---
    textGoodbye = visual.TextStim(win=win, name='textGoodbye',
        text='Obrigado pela sua participação',
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
        components=[textExplanation, key_resp],
    )
    WelcomeScreen.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []
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
        
        # *textExplanation* updates
        
        # if textExplanation is starting this frame...
        if textExplanation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            textExplanation.frameNStart = frameN  # exact frame index
            textExplanation.tStart = t  # local t and not account for scr refresh
            textExplanation.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(textExplanation, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'textExplanation.started')
            # update status
            textExplanation.status = STARTED
            textExplanation.setAutoDraw(True)
        
        # if textExplanation is active this frame...
        if textExplanation.status == STARTED:
            # update params
            pass
        
        # *key_resp* updates
        waitOnFlip = False
        
        # if key_resp is starting this frame...
        if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
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
    if key_resp.keys in ['', [], None]:  # No response was made
        key_resp.keys = None
    thisExp.addData('key_resp.keys',key_resp.keys)
    if key_resp.keys != None:  # we had a response
        thisExp.addData('key_resp.rt', key_resp.rt)
        thisExp.addData('key_resp.duration', key_resp.duration)
    thisExp.nextEntry()
    # the Routine "WelcomeScreen" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler2(
        name='trials',
        nReps=1, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('Components_parameters.xlsx'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial in trials:
        trials.status = STARTED
        if hasattr(thisTrial, 'status'):
            thisTrial.status = STARTED
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "TrainingTrials" ---
        # create an object to store info about Routine TrainingTrials
        TrainingTrials = data.Routine(
            name='TrainingTrials',
            components=[polygonVertical, polygonHorizontal, gabor, keySide],
        )
        TrainingTrials.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        gabor.setContrast(grating)
        # Run 'Begin Routine' code from codeCorKey
        if position[0] > 0:
            correct_ans = "l"
        else:
            correct_ans = "s"
        # create starting attributes for keySide
        keySide.keys = []
        keySide.rt = []
        _keySide_allKeys = []
        # store start times for TrainingTrials
        TrainingTrials.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        TrainingTrials.tStart = globalClock.getTime(format='float')
        TrainingTrials.status = STARTED
        thisExp.addData('TrainingTrials.started', TrainingTrials.tStart)
        TrainingTrials.maxDuration = None
        # keep track of which components have finished
        TrainingTrialsComponents = TrainingTrials.components
        for thisComponent in TrainingTrials.components:
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
        
        # --- Run Routine "TrainingTrials" ---
        thisExp.currentRoutine = TrainingTrials
        TrainingTrials.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 4.0:
            # if trial has changed, end Routine now
            if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
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
                gabor.setPhase(phase, log=False)
            
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
            
            # *keySide* updates
            waitOnFlip = False
            
            # if keySide is starting this frame...
            if keySide.status == NOT_STARTED and tThisFlip >= 1.00-frameTolerance:
                # keep track of start time/frame for later
                keySide.frameNStart = frameN  # exact frame index
                keySide.tStart = t  # local t and not account for scr refresh
                keySide.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(keySide, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'keySide.started')
                # update status
                keySide.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(keySide.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(keySide.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if keySide is stopping this frame...
            if keySide.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > keySide.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    keySide.tStop = t  # not accounting for scr refresh
                    keySide.tStopRefresh = tThisFlipGlobal  # on global time
                    keySide.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'keySide.stopped')
                    # update status
                    keySide.status = FINISHED
                    keySide.status = FINISHED
            if keySide.status == STARTED and not waitOnFlip:
                theseKeys = keySide.getKeys(keyList=['s','l'], ignoreKeys=["escape"], waitRelease=False)
                _keySide_allKeys.extend(theseKeys)
                if len(_keySide_allKeys):
                    keySide.keys = _keySide_allKeys[-1].name  # just the last key pressed
                    keySide.rt = _keySide_allKeys[-1].rt
                    keySide.duration = _keySide_allKeys[-1].duration
                    # was this correct?
                    if (keySide.keys == str(correct_ans)) or (keySide.keys == correct_ans):
                        keySide.corr = 1
                    else:
                        keySide.corr = 0
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
                    currentRoutine=TrainingTrials,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                TrainingTrials.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if TrainingTrials.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in TrainingTrials.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "TrainingTrials" ---
        for thisComponent in TrainingTrials.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for TrainingTrials
        TrainingTrials.tStop = globalClock.getTime(format='float')
        TrainingTrials.tStopRefresh = tThisFlipGlobal
        thisExp.addData('TrainingTrials.stopped', TrainingTrials.tStop)
        # check responses
        if keySide.keys in ['', [], None]:  # No response was made
            keySide.keys = None
            # was no response the correct answer?!
            if str(correct_ans).lower() == 'none':
               keySide.corr = 1;  # correct non-response
            else:
               keySide.corr = 0;  # failed to respond (incorrectly)
        # store data for trials (TrialHandler)
        trials.addData('keySide.keys',keySide.keys)
        trials.addData('keySide.corr', keySide.corr)
        if keySide.keys != None:  # we had a response
            trials.addData('keySide.rt', keySide.rt)
            trials.addData('keySide.duration', keySide.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if TrainingTrials.maxDurationReached:
            routineTimer.addTime(-TrainingTrials.maxDuration)
        elif TrainingTrials.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-4.000000)
        
        # --- Prepare to start Routine "TrialFeedback" ---
        # create an object to store info about Routine TrialFeedback
        TrialFeedback = data.Routine(
            name='TrialFeedback',
            components=[textFeedback, soundFeedback],
        )
        TrialFeedback.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from codeFconditions
        print("correct_ans =", correct_ans, type(correct_ans))
        print("keySide =", keySide, type(keySide))
        
        if correct_ans == keySide.keys:
            score += 1
            text_feedback = f"Correct!\n+1 point\nTotal: {score}"
            win.color = "green"
            sound_feedback = 4000
            volume = 1
            duration = 0.5
        else:
            score += 0
            text_feedback = f"Wrong!\n+0 points\nTotal: {score}"
            win.color = "red"
            sound_feedback = 1000
            volume = 1
            duration = 1
            
        win.flip()
        soundFeedback.setSound(sound_feedback , secs=duration, hamming=True)
        soundFeedback.setVolume(1.0, log=False)
        soundFeedback.seek(0)
        # store start times for TrialFeedback
        TrialFeedback.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        TrialFeedback.tStart = globalClock.getTime(format='float')
        TrialFeedback.status = STARTED
        thisExp.addData('TrialFeedback.started', TrialFeedback.tStart)
        TrialFeedback.maxDuration = None
        # keep track of which components have finished
        TrialFeedbackComponents = TrialFeedback.components
        for thisComponent in TrialFeedback.components:
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
        
        # --- Run Routine "TrialFeedback" ---
        thisExp.currentRoutine = TrialFeedback
        TrialFeedback.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
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
                if tThisFlipGlobal > textFeedback.tStartRefresh + duration-frameTolerance:
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
                    currentRoutine=TrialFeedback,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                TrialFeedback.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if TrialFeedback.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in TrialFeedback.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "TrialFeedback" ---
        for thisComponent in TrialFeedback.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for TrialFeedback
        TrialFeedback.tStop = globalClock.getTime(format='float')
        TrialFeedback.tStopRefresh = tThisFlipGlobal
        thisExp.addData('TrialFeedback.stopped', TrialFeedback.tStop)
        # Run 'End Routine' code from codeFconditions
        win.color="gray"
        soundFeedback.pause()  # ensure sound has stopped at end of Routine
        # the Routine "TrialFeedback" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisTrial as finished
        if hasattr(thisTrial, 'status'):
            thisTrial.status = FINISHED
        # if awaiting a pause, pause now
        if trials.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            trials.status = STARTED
        thisExp.nextEntry()
        
    # completed 1 repeats of 'trials'
    trials.status = FINISHED
    
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
        components=[textGoodbye],
    )
    GoodbyeScreen.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
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
    while continueRoutine and routineTimer.getTime() < 2.0:
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
            pass
        
        # if textGoodbye is stopping this frame...
        if textGoodbye.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > textGoodbye.tStartRefresh + 2.0-frameTolerance:
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
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if GoodbyeScreen.maxDurationReached:
        routineTimer.addTime(-GoodbyeScreen.maxDuration)
    elif GoodbyeScreen.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-2.000000)
    thisExp.nextEntry()
    
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
