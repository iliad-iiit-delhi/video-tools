import kivy
from kivy.app import App

from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.floatlayout import  FloatLayout
from kivy.config import Config
from kivy.uix.slider import Slider
from kivy.uix.image import Image
from kivy.uix.behaviors import ButtonBehavior
from functools import wraps
from kivy.clock import Clock
from kivy.event import EventDispatcher
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Rectangle
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics.texture import Texture
from kivy.uix.dropdown import DropDown
from kivy.uix.spinner import Spinner
from kivy.uix.filechooser import FileChooserListView 
from kivy.uix.popup import Popup
from kivy.uix.checkbox import CheckBox

from array import array
import copy

import mido
import cv2
import numpy as np
import time

import io
from kivy.core.image import Image as CoreImage

Config.set('kivy', 'keyboard_mode', 'systemandmulti')

rate_transfer = 0
message_count = 0
rtmidi = None
midiports = []
outputport = None
vidcap = None
success = False
stopped = 2
image = None
videofilepath = ''
channel = 0
message_type = 'control_change'
num_frames_skip = 0
control = 0


arithmetic_op = 'Sum'
color_space = 'RGB'
crop_type = 'Top left'
crop_x = 100
crop_y = 100

modulo_value = 128
range_min = 0
range_max = 127

# if stopped = 0, then it is not stopped, 1: paused, 2: stopped

# decorator function
def yield_to_sleep(func):
	@wraps(func) # takes a function used in a decorator and adds the functionality of copying over the function name, arguments list, etc. 
	def wrapper(*args, **kwargs):
		gen = func()	
		def next_step(*_): #defining a custom iterator function
			try:
				t = next(gen)  # this executes the part of 'func' (in this case, read_image) before the yield statement and returns control to you
				# i.e in this case it returns run_dict["v"] and stores it in t
			except StopIteration:
				pass
			else:
				# start = time.time()
				Clock.schedule_once(next_step, t)  # having control you can resume func execution after some time
				# print("Process time: " + str(time.time() - start))
				# this executes the part of func after the yield
		next_step()
	return wrapper

# generator function
@yield_to_sleep  # use this decorator to cast 'yield' to non-blocking sleep
def read_video():
	global rate_transfer, success, stopped, message_count, vidcap
	# for i in range(10):
	while(success and stopped == 0):
		# yield run_dict["v"]  # use yield to "sleep"
		# print(stopped)
		# if stopped == 1:
		# 	# wait until the value of stopped changes
		yield rate_transfer
		# value of rate_transfer is returned and the generator state is suspended
		# During the next call the generator resumes where it freezed before and then the value of r is randomly generated. 
		# r = random.randint(10, 90)
		rgb = send_message_midi()
		# g = random.randint(, 40)
		# b = random.randint(30, 50)
		# send_RGB(r)
	print('Stopped:', stopped)
	if not success:
		print('Finished reading video file!')
		# print(vidcap.get(cv2.CAP_PROP_POS_FRAMES))
		stopped = 2 
		vidcap.release()
	elif stopped == 2:
		# rate_transfer = 0
		print('Stopped')
		vidcap.release()

def image_operation():
	global image

	image = perform_arithmetic_opn()

	return image

	# 1. copyMakeBorder
	# replicate = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REPLICATE)
	# reflect = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT)
	# reflect101 = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT_101)
	# wrap = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_WRAP)
	# constant= cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_CONSTANT,value=BLUE)

	# 2. Select image sections
	# select percentx, percenty, corner

	# 3. Bitwise operations
	# Now create a mask of logo and create its inverse mask also
	# img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
	# ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
	# mask_inv = cv2.bitwise_not(mask)

	# # Now black-out the area of logo in ROI
	# img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

	# # Take only region of logo from logo image.
	# img2_fg = cv2.bitwise_and(img2,img2,mask = mask)

	# # Put logo in ROI and modify the main image
	# dst = cv2.add(img1_bg,img2_fg)
	# img1[0:rows, 0:cols ] = dst

	# 4. 


def convert_color_space():
	global color_space, image

	# RGB','Gray','HSV', 'YCrCb','XYZ','HLS'
	image_temp = None
	if color_space == 'Gray':
		image_temp = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	elif color_space == 'HSV':
		image_temp = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	elif color_space == 'YCrCb':
		image_temp = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
	elif color_space == 'XYZ':
		image_temp = cv2.cvtColor(image, cv2.COLOR_BGR2XYZ)
	elif color_space == 'HLS':
		image_temp = cv2.cvtColor(image, cv2.HLS)
	else:
		image_temp = image
	
	return image_temp

def perform_arithmetic_opn():

	# ('Sum','Product','Division','Difference','Power','Gradient','Exponential','Log','Max','Min','Median','Mode')
	global arithmetic_op, image

	if arithmetic_op == 'Sum':
		return np.sum(image)
	elif arithmetic_op == 'Product':
		return np.prod(image)
	# elif arithmetic_op == 'Division':
	# 	return np.prod(image)
	# elif arithmetic_op == 'Difference':
	# 	return np.prod(image)
	# elif arithmetic_op == 'Power':
	# 	return np.prod(image)
	# elif arithmetic_op == 'Gradient':
	# 	return np.prod(image)
	# elif arithmetic_op == 'Exponential':
	# 	return np.prod(image)
	# elif arithmetic_op == 'Log':
	# 	return np.prod(image)
	elif arithmetic_op == 'Max':
		return np.max(image)
	elif arithmetic_op == 'Min':
		return np.min(image)
	elif arithmetic_op == 'Median':
		return np.median(image)
	# elif arithmetic_op == 'Mode':
	# 	return np.max(image)

def crop_image():
	global image, crop_type, crop_x, crop_y
	print('Cropping:', image.shape)
	# let it be bgr image (default) as it would be a big pain to crop color space changed image
	# 'Top left','Top right','Bottom left', 'Bottom right','Center'),
	if crop_x == 100 and crop_y == 100:
		return image
	if crop_type == 'Top left':
		image_temp = image[0:int((crop_x/100)*image.shape[0]),0:int((crop_y/100)*image.shape[1])]
	if crop_type == 'Top right':
		image_temp = image[0:int((crop_x/100)*image.shape[0]),0:int((crop_y/100)*image.shape[1])]
	if crop_type == 'Bottom left':
		image_temp = image[0:int((crop_x/100)*image.shape[0]),0:int((crop_y/100)*image.shape[1])]
	if crop_type == 'Bottom right':
		image_temp = image[0:int((crop_x/100)*image.shape[0]),0:int((crop_y/100)*image.shape[1])]
	if crop_type == 'Center':
		image_temp = image[0:int((crop_x/100)*image.shape[0]),0:int((crop_y/100)*image.shape[1])]
	return image_temp

def build_note(): # does not value mapping operations
	pass


def build_message(value):
	global channel, message_type, control

	# message_types = (
	# 		'control_change',	#channel control value
	# 		'note_on',	#channel note velocity
	# 		'note_off',	#channel note velocity
	# 		'polytouch',	#channel note value
	# 		'program_change',	#channel program
	# 		'aftertouch',	#channel value
	# 		'pitchwheel',	#channel pitch
	# 	)

	if message_type == 'control_change':
		msg = mido.Message('control_change', channel=channel, control=control, value=value)
	elif message_type == 'note_on':
		msg = mido.Message('note_on', note=value) # velocity and time missing
	elif message_type == 'note_off':
		msg = mido.Message('note_off', note=value) # velocity and time missing
	elif message_type == 'polytouch':
		pass
	elif message_type == 'program_change':
		msg = mido.Message('program_change', channel=channel, program=value) # time missing
	# elif message_type == 'polytouch':
	# 	pass
	# elif message_type == 'aftertouch':
	# 	pass
	# elif message_type == 'pitchwheel':
	# 	pass
	return msg


def send_message_midi():
	
	global vidcap, message_count, outputport, success, image, img_change_ev, num_frames_skip, channel, message_type
	# cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
	# print(dt)
	try:
		# if vidcap.isOpened()
		x = vidcap.get(cv2.CAP_PROP_POS_FRAMES)
		# print("cv2.CAP_PROP_POS_FRAMES:", x)
		vidcap.set(cv2.CAP_PROP_POS_FRAMES, x+num_frames_skip)
		success,image = vidcap.read()

		if not success:
			return
		
		image = crop_image()

		image = convert_color_space()

		print(image.shape)

		img_change_ev.update_image(image)

		# note = image_operation()
		note = perform_arithmetic_opn()

		# print(note)
		# print(type(note))
		message_count+=1

		print(note)
		note = int(note)
		note=note%128
		
		# msg = mido.Message('note_on', note=int(note))
		msg = build_message(note)
		# msg = mido.Message('control_change', channel=channel, control=16, value=note)
		outputport.send(msg)
		print('Sent message ', msg, " : " , message_count)
	except cv2.error as e:
		print(str(e))


class Imglayout(FloatLayout):

	def __init__(self,**args):
		super(Imglayout,self).__init__(**args)

		with self.canvas.before:
			Color(0,0,0,0)
			self.rect=Rectangle(size=self.size,pos=self.pos)

		self.bind(size=self.updates,pos=self.updates)

	def updates(self,instance,value):
		self.rect.size=instance.size
		self.rect.pos=instance.pos


class ImageChangeEventDispatcher(EventDispatcher):
	def __init__(self, **kwargs):
		self.register_event_type('on_img_change')
		super(ImageChangeEventDispatcher, self).__init__(**kwargs)

	def update_image(self, value):
		# when update_image is called, the 'on_test' event will be
		# dispatched with the value
		self.dispatch('on_img_change', value)

	def on_img_change(self, *args):
		# print("I am dispatched")
		# dummy, actual work done in UI
		pass

img_change_ev = ImageChangeEventDispatcher()

class kivi_app(App):

	def OnRunButtonPressed(self, instance):
		global vidcap, success, stopped, midiports, outputport,message_count,videofilepath, rtmidi

		try:
			midiports = mido.get_output_names()
			if self.portsdropdown.text not in midiports:
				self.transfer_rate_label.text = 'An unexpected error occured. Please check the midi ports.'
				print(str(e))
			else:
				if outputport is None:
					outputport = rtmidi.open_output(self.portsdropdown.text)
				elif outputport.name != self.portsdropdown.text:
					outputport.close()
					outputport = rtmidi.open_output(self.portsdropdown.text)
				if stopped == 2: 
					message_count = 0
					# vidcap.release()
					vidcap = None #video had been stopped, so start over
					vidcap = cv2.VideoCapture(videofilepath)
					stopped = 0
				elif stopped == 1: # TODO: and the video path has changed 
					stopped = 0 # video had been paused, so do nothing
				success = True
				print(vidcap)
				self.file_selector.disabled = True
				# success,image = self.vidcap.read()
				read_video()
				# self.event.cancel()
				# self.event = Clock.schedule_interval(self.send_next_message_callback,self.rate_transfer)
				# return event
			
		except Exception as e:
			self.transfer_rate_label.text = 'An unexpected error occured. Please check the midi ports.'
			print(str(e))
		

	def OnImageChanged(self, _, __):
		global image, color_space
		# pass
		if color_space == 'Gray':
			buf1 = cv2.flip(image, 0)	
			buf = buf1.tostring()
			image_texture = Texture.create(size=(image.shape[1], image.shape[0]),colorfmt='luminance')
			image_texture.blit_buffer(buf, colorfmt='luminance', bufferfmt='ubyte')
		# elif color_space == 'HSV':
		# 	buf1 = cv2.flip(image, 0)	
		# 	buf = buf1.tostring()
		# 	image_texture = Texture.create(size=(image.shape[1], image.shape[0]),colorfmt='rgb')
		# 	image_texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
		else:
			print(image.shape)
			buf1 = cv2.flip(image, 0)
			buf = buf1.tostring()
			image_texture = Texture.create(size=(image.shape[1], image.shape[0]),colorfmt='bgr')
			image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
		# display image from the texture
		self.im.texture = image_texture

	def OnStopButtonPressed(self, instance):
		# pass
		global stopped
		stopped = 2
		# if self.event != None:
		# 	Clock.unschedule(self.event)
		self.file_selector.disabled = False

	def OnPauseButtonPressed(self, instance):
		# pass
		global stopped
		if stopped == 0:
			stopped = 1
		# if self.event != None:
		# 	Clock.unschedule(self.event)
		self.file_selector.disabled = False

	def OnSliderValueChange(self, instance,value):
		global rate_transfer
		self.transfer_rate_label.text = "Rate of transfer (delay) in secs: " + str(value)
		rate_transfer = value
		# print(value, rate_transfer)
		# self.event = Clock.schedule_interval(self.send_next_message_callback,self.rate_transfer)

		#TODO: interrupt the timer of the sending of messages

	def create_popup(self, instance):
		# create popup layout
		content = BoxLayout(orientation='vertical', spacing=5)
		# popup_width = min(0.95 * Window.width, dp(500))
		self.filechooserpopup = Popup(
			title='Select video file', content=content, size_hint=(0.9, 0.9),
			width=(0.9,0.9))
	
		# create the filechooser
		self.filechooserview = FileChooserListView(
			# path=self.value,
			 size_hint=(1, 1), filters=['*.mp4','*.avi'])
	
		# construct the content
		content.add_widget(self.filechooserview)
		# content.add_widget(SettingSpacer())
	
		# 2 buttons are created for accept or cancel the current value
		btnlayout = BoxLayout(size_hint_y=None, height='40dp', spacing='20dp')
		btn = Button(text='Ok')

		btn.bind(on_release=self.select_video_file_path)
		btnlayout.add_widget(btn)
		btn = Button(text='Cancel')
		btn.bind(on_release=self.filechooserpopup.dismiss)
		btnlayout.add_widget(btn)
		content.add_widget(btnlayout)
	
		# all done, open the popup !
		self.filechooserpopup.open()

	def select_video_file_path(self, instance):
		global videofilepath, stopped
		# videofilepath = self.filechooserview.selection
		print(self.filechooserview.selection)
		if len(self.filechooserview.selection) == 0:
			content = BoxLayout(orientation='vertical', spacing=5)
			popup = Popup(
			title='Please select a video file!', content=content, size_hint=(0.2, 0.2),
			width=(0.2,0.2))
			btn = Button(text='Ok')
			btn.bind(on_release=popup.dismiss)
			content.add_widget(btn)
			popup.open()
		else:
			videofilepath = self.filechooserview.selection[0]
			self.filechooserpopup.dismiss()
			stopped = 2
			self.file_path_text.text = '...' + videofilepath[-50:]
			self.run_button.disabled = False
			self.pause_button.disabled = False
			self.stop_button.disabled = False

	def OnPortChanged(self, spinner, text):
		# pass
		global outputport, midiports, rtmidi
		print("port changed")

		midiports = mido.get_output_names()
		if midiports == []:
			return
		elif self.portsdropdown.text not in midiports:
			self.transfer_rate_label.text = 'An unexpected error occured. Please check the midi ports.'
			print(str(e))
		else:
			if outputport is None:
				outputport = rtmidi.open_output(self.portsdropdown.text)
			elif outputport.name != self.portsdropdown.text:
				outputport.close()
				outputport = rtmidi.open_output(self.portsdropdown.text)

	def OnChannelChanged(self, spinner, text):
		# pass
		global channel
		print("channel number changed")

		channel = int(self.channel_selector.text)

	def OnMessageTypeChanged(self, spinner, text):
		global message_type

		if self.message_type_selector.text != 'control_change' and self.message_type_selector.text != 'program_change':
			self.control_selector.disabled = True
			self.channel_selector.disabled = True
		elif self.message_type_selector.text == 'control_change':
			self.control_selector.disabled = False
			self.channel_selector.disabled = False
		elif self.message_type_selector.text == 'program_change':
			self.control_selector.disabled = True
			self.channel_selector.disabled = False

		print("message type changed")

		message_type = self.message_type_selector.text

	def OnControlChanged(self, spinner, text):
		global control

		print("control value changed")

		control = int(self.control_selector.text)

	def OnFrameSkipChanged(self, instance, text):
		# pass
		global num_frames_skip, vidcap
		print("number of frames to skip changed")

		if self.frames_skip_btn.text == '':
			num_frames_skip = 0
		else:
			num_frames_skip = int(self.frames_skip_btn.text)
		print(num_frames_skip)

		# if vidcap.isOpened():
		# 	vidcap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)

	def OnImgOpnsButtonPressed(self, instance):
		# pass
		# content = BoxLayout(orientation='vertical', spacing=5)
		# popup_width = min(0.95 * Window.width, dp(500))
		if self.imgopnspopup == None:
			content = FloatLayout(size=(400, 400), 
			# orientation='vertical', 
			# spacing=5
			)
			self.imgopnspopup = Popup(
				title='Select image operations', content=content, size_hint=(0.9, 0.9),
				width=(0.9,0.9))

			self.imgopnspopup.arithmetic_opns = Spinner(# default value shown
					text = 'Sum',
					# available values
					values=('Sum',
						'Product',
						# 'Division',
						# 'Difference',
						# 'Power',
						# 'Gradient',
						# 'Exponential',
						# 'Log',
						'Max',
						'Min',
						'Median',
						# 'Mode',
						),
					pos_hint={'x': .53, 'y': .8},
					size_hint=(.3, .075),
					# on_text = self.OnPortChanged,
					)

			self.imgopnspopup.arithmetic_opns_label = Label(text="Arithmetic operation",
							  font_size=14,
							  pos_hint={'x': .2, 'y': .81},
							  size_hint=(.2, .05))

			self.imgopnspopup.color_opns = Spinner(# default value shown
					text = 'RGB',
					# available values
					values=('RGB','Gray','HSV', 'YCrCb','XYZ','HLS'),
					pos_hint={'x': .53, 'y': .7},
					size_hint=(.3, .075),
					# on_text = self.OnPortChanged,
					)

			self.imgopnspopup.color_opns_label = Label(text="Color space",
							  font_size=14,
							  pos_hint={'x': .2, 'y': .71},
							  size_hint=(.2, .05))

			self.imgopnspopup.crop_opns = Spinner(# default value shown
					text = 'Top left',
					# available values
					values=('Top left','Top right','Bottom left', 'Bottom right','Center'),
					pos_hint={'x': .53, 'y': .6},
					size_hint=(.15, .075),
					# on_text = self.OnPortChanged,
					)

			self.imgopnspopup.crop_x = Spinner(text = '100',
						pos_hint={'x': .69, 'y': .6},
						size_hint=(.1, .075),
						values = tuple([str(i) for i in range(100,2,-1)]),
						)

			self.imgopnspopup.crop_y = Spinner(text = '100',
						pos_hint={'x': .8, 'y': .6},
						size_hint=(.1, .075),
						values = tuple([str(i) for i in range(100,2,-1)]),
						)

			self.imgopnspopup.crop_opns_label = Label(text="Crop options, %x, %y",
							  font_size=14,
							  pos_hint={'x': .2, 'y': .61},
							  size_hint=(.2, .05))

			self.imgopnspopup.border_opns = Spinner(# default value shown
					text = "Select border parameters",
					# available values
					values=(),
					pos_hint={'x': .53, 'y': .5},
					size_hint=(.3, .075),
					# on_text = self.OnPortChanged,
					)

			self.imgopnspopup.border_opns_label = Label(text="Add border",
							  font_size=14,
							  pos_hint={'x': .2, 'y': .51},
							  size_hint=(.2, .05))
		
			# 2 buttons are created for accept or cancel the current value
			# btnlayout = BoxLayout(size_hint_y=None, height='40dp', spacing='20dp')
			self.imgopnspopup.save_btn = Button(text='Save', pos_hint={'x': .2, 'y': .075}, size_hint = (.25, .075))
			self.imgopnspopup.save_btn.bind(on_release=self.OnImgOpnsSaved)
			content.add_widget(self.imgopnspopup.save_btn)
			# btn.bind(on_release=self.select_video_file_path)
			# btnlayout.add_widget(btn)
			self.imgopnspopup.cancel_btn = Button(text='Cancel', pos_hint = {'x': .55, 'y': .075}, size_hint = (.25, .075))
			self.imgopnspopup.cancel_btn.bind(on_release=self.imgopnspopup.dismiss)
			content.add_widget(self.imgopnspopup.cancel_btn)
			# btnlayout.add_widget(btn)
			# content.add_widget(btnlayout)
			content.add_widget(self.imgopnspopup.arithmetic_opns_label)
			content.add_widget(self.imgopnspopup.arithmetic_opns)
			content.add_widget(self.imgopnspopup.color_opns_label)
			content.add_widget(self.imgopnspopup.color_opns)
			content.add_widget(self.imgopnspopup.crop_opns_label)
			content.add_widget(self.imgopnspopup.crop_opns)
			content.add_widget(self.imgopnspopup.crop_x)
			content.add_widget(self.imgopnspopup.crop_y)
			content.add_widget(self.imgopnspopup.border_opns_label)
			content.add_widget(self.imgopnspopup.border_opns)
	
		else:
			print('hello')
			self.imgopnspopup.old_state = self.imgopnspopup

			self.imgopnspopup.old_state.arithmetic_opns_text = copy.deepcopy(self.imgopnspopup.arithmetic_opns.text)
			self.imgopnspopup.old_state.color_opns_text = copy.deepcopy(self.imgopnspopup.color_opns.text)
			self.imgopnspopup.old_state.crop_opns_text = copy.deepcopy(self.imgopnspopup.crop_opns.text)
			self.imgopnspopup.old_state.crop_x_text = copy.deepcopy(self.imgopnspopup.crop_x.text)
			self.imgopnspopup.old_state.crop_y_text = copy.deepcopy(self.imgopnspopup.crop_x.text)

			# print(self.imgopnspopup.old_state.arithmetic_opns_text)
			# print(self.imgopnspopup.old_state.color_opns_text)
			self.imgopnspopup.cancel_btn.bind(on_release=self.OnImgOpnsCanceled)
		self.imgopnspopup.open()
		# all done, open the popup !
		

	def OnImgOpnsSaved(self, instance):
		global arithmetic_op, color_space, crop_type, crop_x, crop_y

		arithmetic_op = self.imgopnspopup.arithmetic_opns.text
		# print(arithmetic_op)

		color_space = self.imgopnspopup.color_opns.text
		# print(color_space)

		crop_type = self.imgopnspopup.crop_opns.text
		crop_x = int(self.imgopnspopup.crop_x.text)
		crop_y = int(self.imgopnspopup.crop_y.text)

		self.imgopnspopup.dismiss()


	def OnImgOpnsCanceled(self, instance):

		# assign each option to older state
		# print(self.imgopnspopup.old_state.arithmetic_opns_text)
		# print(self.imgopnspopup.old_state.color_opns_text)
		self.imgopnspopup.arithmetic_opns.text = self.imgopnspopup.old_state.arithmetic_opns_text
		# print(self.imgopnspopup.arithmetic_opns.text)
		self.imgopnspopup.color_opns.text = self.imgopnspopup.old_state.color_opns_text
		# print(self.imgopnspopup.color_opns.text)
		self.imgopnspopup.crop_opns.text = self.imgopnspopup.old_state.crop_opns_text
		self.imgopnspopup.crop_x.text = self.imgopnspopup.old_state.crop_x_text
		self.imgopnspopup.crop_x.text = self.imgopnspopup.old_state.crop_y_text

		self.imgopnspopup.dismiss()


	def OnNoteMapButtonPressed(self, instance):
		# pass
		if self.notemappopup == None:
			content = FloatLayout(size=(400, 400), 
			# orientation='vertical', 
			# spacing=5
			)
			self.notemappopup = Popup(
				title='Select value to note mapping', content=content, size_hint=(0.9, 0.9),
				width=(0.9,0.9))

			self.notemappopup.opns = Spinner(# default value shown
					text = 'Modulo',
					# available values
					values=('Modulo', 'Range fitting'),
					pos_hint={'x': .53, 'y': .8},
					size_hint=(.3, .075),
					# on_text = self.OnPortChanged,
					)

			self.notemappopup.opns_label = Label(text="Select operation",
							  font_size=14,
							  pos_hint={'x': .2, 'y': .81},
							  size_hint=(.2, .05))

			self.notemappopup.modulo_opns = Spinner(# default value shown
					text = '128',
					# available value
					values=tuple([str(i) for i in range(128,1,-1)]),
					pos_hint={'x': .65, 'y': .7},
					size_hint=(.08, .075),
					# on_text = self.OnPortChanged,
					)

			self.notemappopup.modulo_opns_label = Label(text="Modulo value",
							  font_size=14,
							  pos_hint={'x': .2, 'y': .71},
							  size_hint=(.2, .05))

			self.notemappopup.range_min_opns = Spinner(# default value shown
					text = '0',
					# available values
					values=tuple([str(i) for i in range(0,127)]),
					pos_hint={'x': .65, 'y': .6},
					size_hint=(.08, .075),
					)

			self.notemappopup.range_min_opns_label = Label(text="Range minimum",
							  font_size=14,
							  pos_hint={'x': .2, 'y': .61},
							  size_hint=(.2, .05))

			self.notemappopup.range_max_opns = Spinner(# default value shown
					text = '127', 
					# available values
					values = tuple([str(i) for i in range(127,2,-1)]),
					pos_hint={'x': .65, 'y': .5},
					size_hint=(.08, .075),
					# disabled = True,
					)

			self.notemappopup.range_max_opns_label = Label(text="Range maximum",
							  font_size=14,
							  pos_hint={'x': .2, 'y': .51},
							  size_hint=(.2, .05))
		
			
			self.notemappopup.save_btn = Button(text='Save', pos_hint={'x': .2, 'y': .075}, size_hint = (.25, .075))
			self.notemappopup.save_btn.bind(on_release=self.OnNoteMapOpnsSaved)
			content.add_widget(self.notemappopup.save_btn)
			
			self.notemappopup.cancel_btn = Button(text='Cancel', pos_hint = {'x': .55, 'y': .075}, size_hint = (.25, .075))
			self.notemappopup.cancel_btn.bind(on_release=self.notemappopup.dismiss)
			content.add_widget(self.notemappopup.cancel_btn)
		
			content.add_widget(self.notemappopup.opns_label)
			content.add_widget(self.notemappopup.opns)
			content.add_widget(self.notemappopup.modulo_opns_label)
			content.add_widget(self.notemappopup.modulo_opns)
			content.add_widget(self.notemappopup.range_min_opns_label)
			content.add_widget(self.notemappopup.range_min_opns)
			content.add_widget(self.notemappopup.range_max_opns)
			content.add_widget(self.notemappopup.range_max_opns_label)

			self.notemappopup.range_min_opns.disabled = True
			self.notemappopup.range_max_opns.disabled = True
			self.notemappopup.opns.bind(text = self.OnNotemappopupOpnsChanged)
	
		else:
			# print('hello')
			self.notemappopup.old_state = self.notemappopup
			
			self.notemappopup.old_state.opns_text = copy.deepcopy(self.notemappopup.opns.text)
			self.notemappopup.old_state.modulo_opns_text = copy.deepcopy(self.notemappopup.modulo_opns.text)
			self.notemappopup.old_state.range_min_opns_text = copy.deepcopy(self.notemappopup.range_min_opns.text)
			self.notemappopup.old_state.range_max_opns_text = copy.deepcopy(self.notemappopup.range_max_opns.text)

			# print(self.notemappopup.old_state.opns_text)
			# print(self.notemappopup.old_state.modulo_opns_text)
			# print(self.notemappopup.old_state.range_min_opns_text)
			# print(self.notemappopup.old_state.range_max_opns_text)
			
			self.notemappopup.cancel_btn.bind(on_release=self.OnNoteMapOpnsCanceled)
		
		self.notemappopup.open()
	
		# all done, open the popup !

	def OnNotemappopupOpnsChanged(self, spinner, text):
		print('hello')
		if self.notemappopup.opns.text == 'Modulo':
			self.notemappopup.modulo_opns.disabled = False
			self.notemappopup.range_min_opns.disabled = True
			self.notemappopup.range_max_opns.disabled = True
		elif self.notemappopup.opns.text == 'Range fitting':
			self.notemappopup.modulo_opns.disabled = True
			self.notemappopup.range_min_opns.disabled = False
			self.notemappopup.range_max_opns.disabled = False

	def OnNoteMapOpnsSaved(self, instance):
		# pass

		self.notemappopup.dismiss()
		
	def OnNoteMapOpnsCanceled(self, instance):
		# assign each option to older state
		self.notemappopup.opns.text = self.notemappopup.old_state.opns_text
		self.notemappopup.modulo_opns.text = self.notemappopup.old_state.modulo_opns_text
		self.notemappopup.range_min_opns.text = self.notemappopup.old_state.range_min_opns_text
		self.notemappopup.range_max_opns.text = self.notemappopup.old_state.range_max_opns_text
		self.notemappopup.dismiss()

	def on_checkbox_active(checkbox, value):
		if value:
			print('The checkbox', checkbox, 'is active')
		else:
			print('The checkbox', checkbox, 'is inactive')

	def OnRefreshBtnPressed(self, instance):

		global rate_transfer, message_count, rtmidi, midiports, outputport, img_change_ev

		rate_transfer = self.slider1.value
		message_count = 0
		rtmidi = mido.Backend('mido.backends.rtmidi')
		# portmidi = mido.Backend('mido.backends.portmidi')

		# inputport = rtmidi.open_input()
		# outputport = rtmidi.open_output('loopMIDI Port 1', virtual=True) #gives windows error
		# outputport = portmidi.open_output('loopMIDI Port 1', virtual=True) #gives windows error
		try: 
			midiports = mido.get_output_names()
			print(midiports)
			if len(midiports) == 1: #On Windows there is a single ['Microsoft GS Wavetable Synth 0'] available by default
				self.transfer_rate_label.text = 'No MIDI input devices are currently available.'
			else:
				# outputport = rtmidi.open_output(midiports[1]) #try 

				self.transfer_rate_label.text = "Rate of transfer (delay) in secs: " + str(rate_transfer)
				print(cv2.__version__)
				# self.event = None
				self.slider1.bind(value = self.OnSliderValueChange)
				self.run_button.bind(on_press= self.OnRunButtonPressed)
				self.stop_button.bind(on_press = self.OnStopButtonPressed)
				self.pause_button.bind(on_press = self.OnPauseButtonPressed)
				# self.image_box.bind()
				img_change_ev.bind(on_img_change = self.OnImageChanged)
				self.file_selector.bind(on_press = self.create_popup)

				self.portsdropdown.text = midiports[0]
				self.portsdropdown.values = tuple(midiports)
				self.portsdropdown.bind(text = self.OnPortChanged)
				self.message_type_selector.bind(text = self.OnMessageTypeChanged)

				self.channel_selector.bind(text = self.OnChannelChanged)
				self.control_selector.bind(text = self.OnControlChanged)

				self.frames_skip_btn.bind(text = self.OnFrameSkipChanged)

				self.img_opns_btn.bind(on_press=self.OnImgOpnsButtonPressed)
				self.note_mapping_btn.bind(on_press=self.OnNoteMapButtonPressed)

				# self.refresh_btn.bind(on_press=self.OnRefreshBtnPresssed)

		except Exception as e:
			self.transfer_rate_label.text = str(e)
			print(str(e))

	def build(self):

		FLOAT_LAYOUT = FloatLayout(size=(400, 400))

		# global Imagelayout
		self.im = Image()
		image_texture = Texture.create(size=(200,150), colorfmt='rgb')
		size = 200*150*3
		buf = [int(x * 255 / size) for x in range(size)]
		# then, convert the array to a ubyte string
		# buf = b''.join(map(chr, buf))
		arr = array('B', buf)
		# print(type(arr))
		# then blit the buffer
		image_texture.blit_buffer(arr, colorfmt='rgb', bufferfmt='ubyte')
		# image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
		# display image from the texture
		self.im.texture = image_texture
		self.im.allow_stretch = True
		self.im.keep_ratio = True
		self.im.pos_hint = {'center_x': 1.0, 'center_y': 1.0}
		self.im.size_hint = (1.0,1.0)

		self.imgopnspopup = None
		self.file_selector = None
		self.notemappopup = None

		self.image_box = FloatLayout(pos_hint={'center_x': .1, 'center_y': .425},
							 size_hint=(.35, .35)
							 # size_hint=(None, None)
							 )

		self.title_label = Label(text="ILIAD Video Sonification Tools",
						  font_size=25,
						  color=[63/255, 173/255, 168/255, 1],
						  # color=[105, 106, 188, 1],
						  # rgba(240,255,255 ,1 )
						  pos_hint={'x': .4, 'y': .84},
						  size_hint=(.2, .2),
						  bold = True,
						  # italic = True,
						  )


		self.transfer_rate_label = Label(text="Rate of transfer (delay) in secs: 1",
						  font_size=12,
						  pos_hint={'x': .2, 'y': .78},
						  size_hint=(.2, .2))

		self.run_button = Button(text='Run',
							font_size=15,
							pos_hint={'x': .12, 'y': .1},
							size_hint=(.08, .075),
							# disabled = True,
							)

		self.stop_button = Button(text='Stop',
							font_size=15,
							pos_hint={'x': .22, 'y': .1},
							size_hint=(.08, .075),
							# disabled = True,
							)


		self.pause_button = Button(text='Pause',
							font_size=15,
							pos_hint={'x': .32, 'y': .1},
							size_hint=(.08, .075),
							# disabled = True,
							)


		self.slider1 = Slider(min=0.1,
					 max=10, 
					 value=1,
					 step = 0.01,
					 pos_hint={'x': .15, 'y': .77},
					 size_hint=(.3, .1),
					 )

		self.file_selector = Button(text = 'Select video file',
					pos_hint={'x': .17, 'y': .23},
					size_hint=(.2, .075),)

		self.file_path_text = Label(text="No video file selected",
						  font_size=12,
						  pos_hint={'x': .17, 'y': .33},
						  size_hint=(.2, .05))


		self.portsdropdown = Spinner(# default value shown
				# text=midiports[1],
				text = "Select Port",
				# available values
				values=(),
				# just for positioning in our example
				# size_hint=(None, None),
				# size=(100, 44),
				# size_hint=(None, None),
				pos_hint={'x': .65, 'y': .8},
				size_hint=(.25, .07),
				# on_text = self.OnPortChanged,
				)

		self.ports_label = Label(text="Port Number",
						  font_size=12,
						  pos_hint={'x': .47, 'y': .81},
						  size_hint=(.2, .05))

		self.channel_selector = Spinner(# default value shown
				text='0',
				# available values
				values = tuple([str(i) for i in range(16)]),
				# just for positioning in our example
				# size_hint=(None, None),
				# size=(100, 44),
				# size_hint=(None, None),
				pos_hint={'x': .65, 'y': .68},
				size_hint=(.075, .07))

		self.channel_label = Label(text="Channel Number",
						  font_size=12,
						  pos_hint={'x': .47, 'y': .69},
						  size_hint=(.2, .05))

		self.control_selector = Spinner(# default value shown
				text='0',
				# available values
				values = tuple([str(i) for i in range(16)]),
				# just for positioning in our example
				# size_hint=(None, None),
				# size=(100, 44),
				# size_hint=(None, None),
				pos_hint={'x': .82, 'y': .68},
				size_hint=(.075, .07))

		self.control_label = Label(text="Control",
						  font_size=12,
						  pos_hint={'x': .73, 'y': .69},
						  size_hint=(.1, .05))

		message_types = (
			'control_change',	#channel control value
			'note_on',	#channel note velocity
			'note_off',	#channel note velocity
			# 'polytouch',	#channel note value
			'program_change',	#channel program
			# 'aftertouch',	#channel value
			# 'pitchwheel',	#channel pitch
		)

		self.message_type_selector = Spinner(# default value shown
				text=message_types[0],
				# available values
				values = message_types,
				# just for positioning in our example
				# size_hint=(None, None),
				# size=(100, 44),
				# size_hint=(None, None),
				pos_hint={'x': .65, 'y': .56},
				size_hint=(.25, .07))

		self.message_label = Label(text="Message Type",
						  font_size=12,
						  pos_hint={'x': .47, 'y': .57},
						  size_hint=(.2, .05))

		self.img_opns_btn = Button(text = 'Select',
					pos_hint={'x': .67, 'y': .43},
					size_hint=(.2, .07),)

		self.img_opns_label = Label(text="Select image operations",
						  font_size=12,
						  pos_hint={'x': .47, 'y': .44},
						  size_hint=(.2, .05))

		self.frames_skip_btn = TextInput(text = '0',
					pos_hint={'x': .67, 'y': .33},
					size_hint=(.075, .065),
					input_filter='int',
					)

		self.frames_skip_label = Label(text="Number of frames to skip",
						  font_size=12,
						  pos_hint={'x': .47, 'y': .34},
						  size_hint=(.2, .05))

		self.note_mapping_btn = Button(text = 'Select',
					pos_hint={'x': .67, 'y': .22},
					size_hint=(.2, .075),)

		self.note_mapping_label = Label(text="Select value mapping",
						  font_size=12,
						  pos_hint={'x': .47, 'y': .23},
						  size_hint=(.2, .05))

		# self.midi_toggle = CheckBox(
		# 			# text = 'Select',
		# 			pos_hint={'x': .63, 'y': .11},
		# 			size_hint=(.2, .075),
		# 			group = 'group',
		# 			active = True)

		# self.osc_toggle = CheckBox(
		# 			# text = 'Select',
		# 			pos_hint={'x': .70, 'y': .11},
		# 			size_hint=(.2, .075),
		# 			group = 'group',
		# 			active = False)

		self.refresh_btn = Button(text = 'Refresh',
					pos_hint={'x': .67, 'y': .11},
					size_hint=(.2, .075),)

		self.refresh_label = Label(text="Refresh MIDI ports",
						  font_size=12,
						  pos_hint={'x': .47, 'y': .12},
						  size_hint=(.2, .05))

		FLOAT_LAYOUT.add_widget(self.title_label)
		FLOAT_LAYOUT.add_widget(self.transfer_rate_label)
		FLOAT_LAYOUT.add_widget(self.image_box)
		self.image_box.add_widget(self.im)
		FLOAT_LAYOUT.add_widget(self.run_button)
		FLOAT_LAYOUT.add_widget(self.stop_button)
		FLOAT_LAYOUT.add_widget(self.pause_button)
		FLOAT_LAYOUT.add_widget(self.slider1)
		FLOAT_LAYOUT.add_widget(self.file_selector)
		FLOAT_LAYOUT.add_widget(self.file_path_text)
		FLOAT_LAYOUT.add_widget(self.portsdropdown)
		FLOAT_LAYOUT.add_widget(self.ports_label)
		FLOAT_LAYOUT.add_widget(self.message_type_selector)
		FLOAT_LAYOUT.add_widget(self.message_label)
		FLOAT_LAYOUT.add_widget(self.channel_selector)
		FLOAT_LAYOUT.add_widget(self.channel_label)
		FLOAT_LAYOUT.add_widget(self.img_opns_btn)
		FLOAT_LAYOUT.add_widget(self.img_opns_label)
		FLOAT_LAYOUT.add_widget(self.frames_skip_btn)
		FLOAT_LAYOUT.add_widget(self.frames_skip_label)
		FLOAT_LAYOUT.add_widget(self.note_mapping_btn)
		FLOAT_LAYOUT.add_widget(self.note_mapping_label)

		FLOAT_LAYOUT.add_widget(self.refresh_btn)
		FLOAT_LAYOUT.add_widget(self.refresh_label)

		# FLOAT_LAYOUT.add_widget(self.midi_toggle)
		# FLOAT_LAYOUT.add_widget(self.osc_toggle)
		FLOAT_LAYOUT.add_widget(self.control_selector)
		FLOAT_LAYOUT.add_widget(self.control_label)

		self.run_button.disabled = self.pause_button.disabled = self.stop_button.disabled = True

		self.refresh_btn.bind(on_press=self.OnRefreshBtnPressed)

		global rate_transfer, message_count, rtmidi, midiports, outputport, img_change_ev

		rate_transfer = self.slider1.value
		message_count = 0
		rtmidi = mido.Backend('mido.backends.rtmidi')
		# portmidi = mido.Backend('mido.backends.portmidi')

		# inputport = rtmidi.open_input()
		# outputport = rtmidi.open_output('loopMIDI Port 1', virtual=True) #gives windows error
		# outputport = portmidi.open_output('loopMIDI Port 1', virtual=True) #gives windows error
		try: 
			midiports = mido.get_output_names()
			print(midiports)
			if len(midiports) == 1: #On Windows there is a single ['Microsoft GS Wavetable Synth 0'] available by default
				self.transfer_rate_label.text = 'No MIDI input devices are currently available.'
			else:
				# outputport = rtmidi.open_output(midiports[1]) #try 
				self.transfer_rate_label.text = "Rate of transfer (delay) in secs: " + str(rate_transfer)
				print(cv2.__version__)
				# self.event = None
				self.slider1.bind(value = self.OnSliderValueChange)
				self.run_button.bind(on_press= self.OnRunButtonPressed)
				self.stop_button.bind(on_press = self.OnStopButtonPressed)
				self.pause_button.bind(on_press = self.OnPauseButtonPressed)
				# self.image_box.bind()
				img_change_ev.bind(on_img_change = self.OnImageChanged)
				self.file_selector.bind(on_press = self.create_popup)

				self.portsdropdown.text = midiports[0]
				self.portsdropdown.values = tuple(midiports)
				self.portsdropdown.bind(text = self.OnPortChanged)
				self.message_type_selector.bind(text = self.OnMessageTypeChanged)

				self.channel_selector.bind(text = self.OnChannelChanged)
				self.control_selector.bind(text = self.OnControlChanged)

				self.frames_skip_btn.bind(text = self.OnFrameSkipChanged)

				self.img_opns_btn.bind(on_press=self.OnImgOpnsButtonPressed)
				self.note_mapping_btn.bind(on_press=self.OnNoteMapButtonPressed)

		except Exception as e:
			self.transfer_rate_label.text = str(e)
			print(str(e))
		
		return FLOAT_LAYOUT

if __name__ == '__main__':

	kivi_app().run()

	# TODO: need to put the following stuff on a different thread