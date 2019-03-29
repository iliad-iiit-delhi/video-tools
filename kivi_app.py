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

from array import array

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
	print(stopped)
	if not success:
		print('Finished reading video file!')
	elif stopped == 2:
		# rate_transfer = 0
		print('Stopped')
		# vidcap = None

def send_message_midi():
	
	global vidcap, message_count, outputport, success, image, img_change_ev
	# cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
	# print(dt)
	try:
		success,image = vidcap.read()

		img_change_ev.update_image(image)

		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		# print('Read a new frame with shape: ', gray.shape)
		message_count+=1

		# for i in range(5):
		note = np.mean(gray)
		# print(note)
		# note = np.mean(note)
		# print(note)
		note=note//2
		# print(note)
		# print(type(note))
		# msg = mido.Message('note_on', note=int(note))
		msg = mido.Message('control_change', channel=1-1, control=16, value=int(note))
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
		global vidcap, success, stopped, midiports, outputport,message_count,videofilepath

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
		global image
		# pass
		buf1 = cv2.flip(image, 0)
		buf = buf1.tostring()
		image_texture = Texture.create(size=(image.shape[1], image.shape[0]), colorfmt='bgr')
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
		btnlayout = BoxLayout(size_hint_y=None, height='40dp', spacing='40dp')
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
		self.image_box = FloatLayout(pos_hint={'center_x': .1, 'center_y': .425},
							 size_hint=(.35, .35)
							 # size_hint=(None, None)
							 )


		self.transfer_rate_label = Label(text="Rate of transfer (delay) in secs: 1",
						  font_size=15,
						  pos_hint={'x': .2, 'y': .8},
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
					 step = 1,
					 pos_hint={'x': .15, 'y': .8},
					 size_hint=(.3, .1),
					 )

		self.file_selector = Button(text = 'Select video file',
					pos_hint={'x': .17, 'y': .23},
					size_hint=(.2, .075),)

		self.file_path_text = Label(text="No video file selected",
						  font_size=12,
						  pos_hint={'x': .17, 'y': .33},
						  size_hint=(.2, .05))

		FLOAT_LAYOUT.add_widget(self.transfer_rate_label)
		# FLOAT_LAYOUT.add_widget(text_box)
		FLOAT_LAYOUT.add_widget(self.image_box)
		self.image_box.add_widget(self.im)
		FLOAT_LAYOUT.add_widget(self.run_button)
		FLOAT_LAYOUT.add_widget(self.stop_button)
		FLOAT_LAYOUT.add_widget(self.pause_button)
		FLOAT_LAYOUT.add_widget(self.slider1)
		FLOAT_LAYOUT.add_widget(self.file_selector)
		FLOAT_LAYOUT.add_widget(self.file_path_text)

		self.run_button.disabled = self.pause_button.disabled = self.stop_button.disabled = True

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

				print(cv2.__version__)
				# self.event = None
				self.slider1.bind(value = self.OnSliderValueChange)
				self.run_button.bind(on_press= self.OnRunButtonPressed)
				self.stop_button.bind(on_press = self.OnStopButtonPressed)
				self.pause_button.bind(on_press = self.OnPauseButtonPressed)
				# self.image_box.bind()
				img_change_ev.bind(on_img_change = self.OnImageChanged)
				self.file_selector.bind(on_press = self.create_popup)

				self.portsdropdown = Spinner(# default value shown
				text=midiports[1],
				# available values
				values=tuple(midiports),
				# just for positioning in our example
				# size_hint=(None, None),
				size=(100, 44),
				pos_hint={'x': .55, 'y': .6},
				size_hint=(.3, .1))

				FLOAT_LAYOUT.add_widget(self.portsdropdown)


		except Exception as e:
			self.transfer_rate_label.text = str(e)
			print(str(e))
		
		return FLOAT_LAYOUT

	def calculate(self, *args):
		print(args)


if __name__ == '__main__':

	kivi_app().run()

	# TODO: need to put the following stuff on a different thread