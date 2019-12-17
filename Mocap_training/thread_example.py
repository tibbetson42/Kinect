
# import the necessary packages
import datetime
import time
import cv2
import pickle
import os
import math
import numpy 				as np
import vision_CONFIG 		as cf
import matplotlib.pyplot 	as plt
from threading 				import Thread
from collections 			import OrderedDict

###############################################################################
def checkForOverwrite(bool, message = False):
	""" break, allows user to indicate they intend to overwrite a file.
		Args:
			bool: (T/F)
				Flag of some sort. Often something like os.path.exists(file)
				to determine if it warrants a warning.
			message: (string)
				Message to print to screen if an overwrite warning occurs.
	"""
	if bool:
		print('WARNING: YOU ARE ABOUT TO OVERWRITE DATA OR FILE THAT ALREADY EXISTS')
		if message:
			print(message)
		print('Press "Y" to continue, anything else to cancel')
		cv2.namedWindow('Overwrite Warning')
		if cv2.waitKey(0)== ord('y'):
			print('overwriting...')
			print('')
			cv2.destroyWindow('Overwrite Warning')
			return 1
		else:
			print('choosing to not overwrite the file')
			cv2.destroyWindow('Overwrite Warning')
			print('')
			return 0
	else:
		return 1

###############################################################################
class newVideoRecorder:
	""" Class: newVideoRecorder(camera,fps,window = 'Video', threaded = True)
		Creates an object for recording video. Saves frames into a list until
		the save function is called. Built to work with MovingObject class.
		Args:
			camera: (int)
				which camera to use. If only one camera connected, set to 0.
				2+ cameras, 1 = 2nd camera, 2 = 3rd camera, etc.
			fps: (int/float)
				this is your DESIRED fps. the actual video will be slightly
				slower or faster. when you call stop(), the true FPS will be displayed,
				and can be seen after by calling .FPStracker.fps()
			window: (str) default 'Video'
				string name for cv2 window to display live feed on.
			threaded (bool) default True
				if true, puts video stream (vs) in a separate thread, and also
				records frames in seperate thread. This generally leads to better
				FPS accuracy, but slows down the processing speed of the main
				script in some older computers (the split threads will be fine, image
				processing gets slowed in the central thread)
		example use:
			(initialize MovingObjects and such)
			recorder = newVideoRecorder(0,60)
			recorder.showFeedandWait(ObjectList)
			recorder.start(); print('press q to stop recording')
			while:
				# do any proccessing with recorder.get
				if q press:
					recorder.stop()
					break
			recorder.stop()
			recorder.saveVideo(path,name,format)



	"""
	def __init__(self,camera,fps,window = 'Video',threaded = True):
		self.threaded = threaded
		if self.threaded:
			self.vs = WebcamVideoStream(camera)
			self.vs.start()
			self.thread = Thread(target = self.record)
		else:
			self.vs = cv2.VideoCapture(camera)

		self.FPStracker = FPStracker()
		re,img = self.vs.read()
		self.Video = [img]
		self.desiredFPS = fps
		self.timePerFrame = 1/float(fps)
		self.recording = False
		self.window = window
		self.stopped = False

	def get(self):
		""" Function to pull most recent timestamp and image
		Args: (self)
		Returns:
			new (bool)
				flag indicating if the image is different than last get() call
			img (np.array)
				image frame, BGR format
			time (float)
				time stamp of img
		"""
		if not self.threaded:
			self.record()
		img = self.Video[-1]
		time = self.timestamps[-1]
		if self.newAvailable:
			new = True
			self.newAvailable = False
			return new, img, time
		else:
			new = False
			return new, img, time

	def start(self):
		self.timestamps = [datetime.datetime.now()]
		self.stopped = False
		""" Begin adding frames to video sequence. Call .stop() to finish recording. """
		print('beginning to record')
		self.recording = True
		self.FPStracker.start()
		if self.threaded:
			self.thread.start()
		else:
			print('warning, this video recorder is not threaded')
			print('you will still need to call .record() ')
		self.newAvailable = False
		return self

	def record(self):
		""" Function that adds frames to video list. All internal. If self.threaded,
		on self.start() needs to be called. If self.threaded = False (due to
		speed problems mentioned in class.__doc__), then script needs to call
		record every iteration. A good failsafe is below.
		ex)
			recorder.start()
			while bool:
				if not recorder.threaded:
					recorder.record()
				# do stuff with recorder.get()
		 """
		while True:
			if not self.recording:
				break
			#print('hal')
			elapsed = (datetime.datetime.now() - self.timestamps[-1]).total_seconds()
			if elapsed < self.timePerFrame:
				pass
			else:
				#print(len(self.Video))
				ret,frame = self.vs.read()
				if ret:
					self.timestamps.append(datetime.datetime.now())
					self.Video.append(frame)
					self.FPStracker.update()
					self.newAvailable = True
					if not self.threaded:

						return
				else:
					print('error: camera failed to capture image')
					print('canceling recording session')
					self.stop()
		#print('\n     recording loop ended, returning to main')
		self.vs.stop()
		return

	def stop(self):
		if self.stopped:
			return
		""" End recording. Also stop the FPS tracker, and any video threads.
		"""
		print('stopping recording')
		self.recording = False
		try:
			self.FPStracker.stop()
			print('     fps is {}'.format(self.FPStracker.fps()) )
		except:
			pass
		if self.threaded:
			self.vs.stop()
			try:
				print('     joining video thread')
				self.thread.join(3.0)
				print('     thread joined')
			except:
				print('    warning: thread not started so join() does nothing')

		else:
			self.vs.release()
		#cv2.destroyWindow(self.window)
		print('     stopped successfully\n')
		self.stopped = True
		return

	def saveVideo(self,path,filename,filetype):
		""" saves the video frames stored in self.video as a video to
		specified location. After saving, ensure
		Args:
			path (str)
				path to save to. Recommend use global path C://etc, not reference
				to current path.
			filename (str)
				name of video file
			filetype (str)
				video format extension. .avi, .mp4 etc.
				NOTE: IF FORMAT CHANGES, CONFIG.FOURCC NEEDS TO CHANGE AS WELL
				at time of writing, it is set for .avi.

		"""
		print('saving video...')
		if self.recording:
			print('warning, video was still recording, please call .stop() in main script to end more consistently.')
			self.stop()
		frame = self.Video[-1]
		(frame_height,frame_width) = frame.shape[0:2]
		savepath = path + filename + filetype
		out = cv2.VideoWriter(savepath,cf.FOURCC,self.desiredFPS,(frame_width,frame_height))
		for i,frame in enumerate(self.Video):
			out.write(frame)
		print('     video saved\n')
		if self.threaded: # guarantee thread stopped
			self.vs.stop()
		self.stop()

	def showFeedandWait(self,ObjectList = None):
		""" Function for setting up recording. Will wait until R is pressed
		to proceed. THIS IS NOT A RECORDING. Used for feedback for the user
		before recording starts.

		Args:
			ObjectList (List of MovingObjects), default None
				Each MovingObject has 2 trackers associated with them. Feeding
				a list of objects means showFeedandWait() will also wait
				for adjustment of the trackers in addition to live video feed.

		generally good setup:
			recorder.showFeedandWait() #to indicate user ready
			recorder.start()

		"""

		print('Adjust trackers/camera, press "R" when ready to continue')
		print('Or, press "Q" to cancel video capture')
		timestamp = datetime.datetime.now()
		while True:
			#wait 1/fps seconds to take new picture
			elapsed = (datetime.datetime.now()-timestamp).total_seconds()
			if elapsed < self.timePerFrame:
				pass
			else:
				timestamp = datetime.datetime.now()
				ret,frame = self.vs.read()
				if ret:
					if ObjectList is not None:
						for object in ObjectList:
						    frame = object.updateTrackers(frame)
					cv2.imshow(self.window,frame)
					key = cv2.waitKey(1)
					if key == ord('q'):
						print('     Q: canceling session\n')
						self.stop()
						quit()
					elif key == ord('r'):
						print('     R: moving on\n')
						break
				else:
					print('error: camera failed to capture image')
					print('canceling recording session')
					if self.threaded:
						vs.stop()
					quit()
		return


###############################################################################
def writeMetadata(path,filename,filetype,ObjectList,VideoRecorder = None):
	"""Creates metadata dictionary. DOES NOT SAVE IT. Function works paired
	with MovingObjects and VideoRecorder classes. Requires the video to be saved
	already.
	Args:
		path (str)
			path video is in. Recommend use global path C://etc, not reference
			to current path.
		filename (str)
			name of video file
		filetype (str)
			video format extension. .avi, .mp4 etc.
			NOTE: IF FORMAT CHANGES, CONFIG.FOURCC NEEDS TO CHANGE AS WELL
			at time of writing, it is set for .avi.
		ObjectList (List of MovingObjects)
			Records relevant data about each MovingObject in the metadata file.
		VideoRecorder (newVideoRecorder object), default None
			Pass this if its available. Will result in more accurate FPS data.
			Without this, no guarantee on accuracy of FPS measure.
	Returns:
		metadata (ordered dictionary)
			dictionary of various parameters. Used when writing data and saving data.
			See writeData() and saveData(). printMetadata(metadata) will print it all out.
	"""
	print('writing metadata, for saving to {}'.format(path+filename+'.pickle'))
	now = datetime.datetime.now() # current date and time
	metadata = OrderedDict()
	metadata['Path'] = path
	metadata['Filename'] = filename
	metadata['Format'] = filetype
	metadata['datetime'] = now
	v = cv2.VideoCapture(path+filename+filetype)
	metadata['Frames'] = v.get(cv2.CAP_PROP_FRAME_COUNT)

	if VideoRecorder is not None:
		fps = VideoRecorder.FPStracker.fps() # if you have a more accurate measure
	else:
		try:
			fps = loadData(path,filename)[0]['FPS']
		except:
			fps = None
		if fps is not None:
			pass
		else:
			fps = v.get(cv2.CAP_PROP_FPS) # trusting camera FPS
	metadata['FPS'] = fps
	metadata['Length'] = metadata['Frames']/metadata['FPS']
	metadata['Resolution'] = [v.get(3),v.get(4)]
	v.release()
	# Save the object description (not the x,y,theta data: no processing yet)
	# and tracker coordinates for every object
	metadata['Num Objects'] = len(ObjectList)
	for i,object in enumerate(ObjectList):
		key = "object{}".format(i)
		t1 = object.Tracker1
		t2 = object.Tracker2
		coord1 = [t1.x,t1.y,t1.w,t1.h,t1.ang]
		coord2 = [t2.x,t2.y,t2.w,t2.h,t2.ang]
		metadata[key+'_ID'] = object.ID
		metadata[key+'_profile'] = object.IDprofile
		metadata[key+'_Tracker1_Coords'] = coord1
		metadata[key+'_Tracker1_BGR_range']  = t1.bgrRange
		metadata[key+'_Tracker2_Coords'] = coord2
		metadata[key+'_Tracker2_BGR_range']  = t2.bgrRange
	return metadata

###############################################################################
def writeData(metadata,ObjectList):
	""" Creates a dictionary to store time,x,y,theta of each object in ObjectList
	Args:
		metadata (ordered dictionary)
			use metadata from writeMetadata() here. will automatically try to
			load existing data from that path, then add any new data, or overwrite.
			Recall that MovingObjects have both Live and Post-processed data.
		ObjectList (list of MovingObjects)
			takes the data from each MovingObject in this list for saving.
	Returns:
		data (ordered dictionary)
			dictionary with time,x,y,theta information on each object.
	"""
	path = metadata['Path']
	filename = metadata['Filename']
	filetype = metadata['Format']
	print('creating data to save to {}'.format(path+filename+'.pickle'))
	try:
		data = loadData(path,filename)[1]
	except:
 		data = __initializeData()
	data['Num Objects'] = len(ObjectList)
	if not data['Num Objects'] == metadata['Num Objects']:
		print('warning: mistmatch between number of objects in metadata and ObjectList')
		print('you may want to double check before any overwriting')
		time.sleep(2)

	message = 'old data last saved on ' + data['Time_Written_POST']
	# save the post-processing data if indicate overwrite is ok
	if checkForOverwrite(data['Saved_POST'],message):
		for i,object in enumerate(ObjectList):
			key = "object{}".format(i)
			data[key+'_ID'] = object.ID
			data[key+'_time_POST'] = object.PostData.Time
			data[key+'_x_POST'] = object.PostData.X
			data[key+'_y_POST'] = object.PostData.Y
			data[key+'_theta_POST'] = object.PostData.Theta
		data['Saved_POST'] = True
		data['Time_Written_POST'] = datetime.datetime.now().strftime('%m-%d-%Y, %H:%M')
	else:
		print('old data kept, new data not written')
	# ONLY save live data if it was never saved before
	if not data['Saved_LIVE']:
		for i,object in enumerate(ObjectList):
			data[key+'_ID'] = object.ID
			data[key+'_time_LIVE'] = object.LiveData.Time
			data[key+'_x_LIVE'] = object.LiveData.X
			data[key+'_y_LIVE'] = object.LiveData.Y
			data[key+'_theta_LIVE'] = object.LiveData.Theta
		data['Saved_LIVE'] = True
		data['Time_Written_LIVE'] = datetime.datetime.now().strftime('%m-%d-%Y, %H:%M')
	return data

def saveData(metadata,data = False,savepath = None):
	""" pickles metadata and data into a pickle file with same name as video.
	Args:
		metadata (ordered dictionary):
			from writeMetadata(...)
		data (ordered dictionary):
			from(writeData(metadata,ObjectList))
		savepath (str) default None
			if you have a specific directory (want to save somewhere new, maybe
			different than video)

	"""
	filename = metadata['Filename']
	meta = metadata.copy() # dont overwrite variable
	if savepath is not None: #if saving to new path, change the metadata entry
		meta['Path'] = savepath
	if not data:
		data = __initializeData()
	savepath = meta['Path']
	allVideoData = [meta,data]
	with open(savepath+filename+".pickle","wb") as handle:
		pickle.dump(allVideoData,handle,protocol = pickle.HIGHEST_PROTOCOL)
	print('data saved to {}'.format(savepath+filename+".pickle"))
	return

###############################################################################
def loadData(path,filename):
	""" load the data and metadata from specified path. Some error warnings
	if cannot find file, or some of the data/metadata is missing from what is
	expected
	Args:
		path (str)
			path to pickle file
		filename(str)
			name of pickle file (no extension)
	Returns:
		metadata(OrderedDict):
			returns False if metadata missing
		data(OrderedDict):
			returns output of __initializeData() if metadata or data had problems,
			otherwise loads data dictionary from pickle file per norm.
	"""
	filepath = path + filename
	try:
		#print('try1')
		with open(filepath+".pickle","rb") as handle:
			allVideoData = pickle.load(handle)
		try:
			metadata = allVideoData[0]
			data = allVideoData[1]
		except:
			metadata = allVideoData
			data = __initializeData()
			print("WARNING")
			print("warning: no data attached to metadata, initializing empty set")
			time.sleep(1)
		return metadata,data
	except:
		print('no file {} exists yet'.format(filepath+".pickle"))
		print('if writeMetadata has already been used, be sure to save it with saveData()')
		time.sleep(1)
		metadata = False
		return metadata,__initializeData()

###############################################################################
def recreateAnalysis(path,filename, dataset = 'POST',save = False, color = (0,255,0),location = (0,0)):
	""" Loads metadata and data from path, loads the connected video. Plays a visual representation of
	the image processing on the data.
	Args:
		path (str)
			save path
		filename (str)
			name of file, no extension
		dataset (str) default 'POST'
			show video for live data or for post-processed data. 'LIVE' is other option.
		save (bool) default False
			if you want to save the video with the annotations
		color (3x1 BGR int tuple) default (0,255,0)
			color of annotations. default green.
		location (2x1 int tuple) default (0,0)
		 	screen location of video to appear (just cv2.namedWindow, cv2.moveWindow(location))
	"""
	print('going to recreate the recording, press anything when ready')
	cv2.namedWindow('Video'); cv2.moveWindow('Video',location[0],location[1])
	cv2.waitKey(0)
	R = cf.OBJECT_RADIUS
	meta,data = loadData(path,filename)
	fps = meta['FPS']
	filetype = meta['Format']
	vid = []
	while True: #replay loop
		VidCap = cv2.VideoCapture(path+filename+filetype)
		i = 0
		while True: #video loop
			re,img = VidCap.read()
			#print(len(vid))
			#cv2.imshow('tmp',img)
			if re:
				for k in range(0,meta['Num Objects']):
					key = 'object{}'.format(k)
					cx = data[key+'_x_'+dataset][i]; cy = data[key+'_y_'+dataset][i]; ang = data[key+'_theta_'+dataset][i]
					img = cv2.circle(img,(cx,cy),4,(200,200,255),-1)
					img = cv2.circle(img,(cx,cy),R,color)
					img = cv2.putText(img,key+': '+data[key+'_ID'],(cx,cy-R),cv2.FONT_HERSHEY_SIMPLEX,0.5,(color))
					pnt1 = (int(cx+R*math.cos(ang)),int(cy-R*math.sin(ang)))
					img = cv2.arrowedLine(img,(cx,cy),pnt1,color)
				cv2.imshow('Video',img)
				vid.append(img)
				i += 1
				if cv2.waitKey(int(1000/fps))== ord('q'):
					VidCap.release()
					print('ending early, not saving')
					save = False
					break
			else:
			    break
		print('press r to replay, or anything else to quit')
		if cv2.waitKey(0) == ord('r'):
		    print('playing processed video again')
		else:
		    break
	VidCap.release()
	if save:
		im = vid[-1]
		(frame_height,frame_width) = im.shape[0:2]
		videoPath = path+filename + '_EVAL_' + dataset + filetype
		out = cv2.VideoWriter(videoPath, cf.FOURCC, fps, (frame_width,frame_height))
		for i,frame in enumerate(vid):
			out.write(frame)
		print('processed video file saved')
	return

###############################################################################
def printMetadata(metadata):
	""" print metadata to terminal
	Args:
		metadata (OrderedDict)
	Returns:
		T/F for if the metadata was intact
	"""
	if metadata:
		for x in metadata:
		    print(x+': {}'.format(metadata[x]) )
		return True
	else:
		print('metadata empty')
		print('unable to find metadata at this location')
		print('')
		return False

###############################################################################
def plot_X_Y_Theta(time,x,y,theta,object = 0,title = '',units = 'pixels'):
	""" Shortcut to plot x,y,theta vs time. Use pull_Time_X_Y_Theta(data,objnum)
	to pull time, x ,y, theta from a data OrderedDict
	Args:
		time,x,y,ang:
			lists of data, (float,int/float,int/float,float)
		title (str):
			title of plot
		units (str):
			doesn't change anything numerically, just a label of x axis units.
	returns matplotlib.pyplot figure
	"""
	fig,axs = plt.subplots(2,1)
	axs[0].plot(time,y)
	axs[0].plot(time,x)
	axs[1].plot(time,theta)
	axs[0].legend(['y','x'])
	axs[1].legend(['theta'])
	axs[0].set(xlabel = 'time(s)',  ylabel = 'position ('+ units + ')', title = title)
	axs[1].set(xlabel = 'time(s)', ylabel = 'angle (rad)')
	axs[1].grid()
	axs[0].grid()
	return fig


def convertPixelsToMetric(X,Y):
	"""convert pixel data to metric data See camera config for ratio calculating.
	Args:
		x(int list): pixel data
		y(int list): pixel data
	Returns:
		x,y as metric float list.
	"""
	x = np.multiply(X,cf.CAM_PIXELS_TO_X)
	y = np.multiply(Y,cf.CAM_PIXELS_TO_Y)
	return x,y


###############################################################################
def pull_Time_X_Y_Theta(data,object = 0, dataset = 'POST'):
	""" Pull time,x,y,theta from data OrderedDict.
	Args:
		data(OrderedDict)
			data from pickle file that you loaded
		object (int) default 0
			which object of the dataset do we want to see? default 0,
			since at least the 1 exists
		dataset (str) default 'POST'
			which dataset ('LIVE' or 'POST'-processed) you want to pull from.
	returns time,x,y,ang
		all 4 return as False if failure to load

	"""
	if data:
		if data['Saved_'+dataset]:
			key = 'object' + str(object)
			time = data[key+'_time_'+dataset]
			x = data[key+'_x_'+dataset]
			y = data[key+'_y_'+dataset]
			ang = data[key+'_theta_'+dataset]
			return time,x,y,ang
		else:
			print('warning: data loaded, but empty. Run video analysis')
			return False,False,False,False
	else:
		print('error: failed to load data')
		return False,False,False,False

###############################################################################
def __initializeData():
	"""internal function, makes empty data set"""
	data = OrderedDict()
	data['Saved_LIVE'] = False
	data['Saved_POST'] = False
	data['Time_Written_POST'] = datetime.datetime.now().strftime('%m-%d-%Y, %H:%M')
	data['Time_Written_LIVE'] = datetime.datetime.now().strftime('%m-%d-%Y, %H:%M')
	return data

###############################################################################
class FPStracker:
	""" used internally newVideoRecorder, but can be used to track iterations
	per second of any while loop
	ex)
		fpstracker = FPStracker()
		fpstracker.start()
		while:
			do stuff
			fpstracker.update()
		fpstracker.stop()
		fps = fpstracker.fps()
		timeelapsed = fpstracker.elapsed()

	 """
	def __init__(self):
		# store the start time, end time, and total number of frames
		# that were examined between the start and end intervals
		self._start = None
		self._end = None
		self._numFrames = 0

	def start(self):
		# start the timer
		self._start = datetime.datetime.now()
		return self

	def stop(self):
		# stop the timer
		self._end = datetime.datetime.now()

	def update(self):
		# increment the total number of frames examined during the
		# start and end intervals
		self._numFrames += 1

	def elapsed(self):
		# return the total number of seconds between the start and
		# end interval
		return (self._end - self._start).total_seconds()

	def fps(self):
		# compute the (approximate) frames per second
		return round(self._numFrames / self.elapsed(),2)

###############################################################################
# Builds video recording in a separate thread
class WebcamVideoStream:
	""" Class for opening a webcam as a seperate thread. Useful to prevent
	bottleneck of communication between camera and computer.
	Stores the most recent camera image, so that a main thread can pull directly
	from memory rather than communicating with camera. Used in newVideoRecorder
	if threading is enabled. Format of reading is very similar to cv2.VideoCapture

	ex)
	webcam = WebcamVideoStream(camera = 0)
	webcam.start()
	while:
		ret,frame = webcam.read()
		do stuff
	webcam.stop()

	"""
	def __init__(self, camera=0):
		self.stream = cv2.VideoCapture(camera)
		if not self.stream.isOpened():
			print("ERROR! video not open. Check camera connection")
			print("Exiting program")
			exit()
		print('Webcam thread opened')
		(self.grabbed, self.frame) = self.stream.read()
		self.stopped = False
		return

	def start(self):
		self.thread = Thread(target=self.update, args=())
		self.thread.start()
		return self

	def update(self):
		while True:
			if self.stopped:
			    break
			self.grabbed,self.frame = self.stream.read()

	def read(self):
		return self.grabbed,self.frame

	def stop(self):
		self.stream.release()
		self.stopped = True
		self.thread.join(2.0)
