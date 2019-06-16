#!/bin/python3

import sys
import time
import threading

import cv2
import numpy as np

import Person


class PersonCounter(object):

    font = cv2.FONT_HERSHEY_SIMPLEX
    kernelOp = np.ones((3,3),np.uint8)
    kernelCl = np.ones((11,11),np.uint8)

    def __init__(self, input_source, width=None, height=None, display_window=True,
                 save_output=False, output_filename_prefix='output',
                 algorithm_params=None, mqtt_fn=None):
        self.input_source = input_source
        self.save_output = save_output
        self.display_window = display_window
        self.out_width = None
        self.out_height = None

        self.mqtt_fn = mqtt_fn if mqtt_fn else lambda m: True

        self._capture_lock = threading.Lock()

        self.params = dict(
            n_frames = 1,                  # analyze every Nth frame
            n_first_frames = 15,           # analyze first N frames to create a backgroud model
            update_bg_every_n_frames = 50, # update the background model every N frames
            threshold = 30,                # threshold for foreground mask
            area_threshold = 1700,         # threshold of the area to treat the object as a person
            light_threshold = 90,          # the difference in luminance(YCrCb)/value(HSV) that will be treated as light change
            max_person_age = 15,           # the number of repeated position coordinates to be treated as standing person
        )
        if algorithm_params:
            self.params.update(algorithm_params)

        self.cap = cv2.VideoCapture(self.input_source);    # open the video stream from a file a device or

        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.out_width = width
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.out_height = height

        self.fps    = self.cap.get(cv2.CAP_PROP_FPS)
        self.width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)) # optional

        self.outStream1 = None
        self.outStream2 = None
        if self.save_output:
            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.outStream1 = cv2.VideoWriter(output_filename_prefix+'_1.avi', fourcc, self.fps, (2*self.width+6, 2*self.height+6))
            self.outStream2 = cv2.VideoWriter(output_filename_prefix+'_2.avi', fourcc, self.fps, (  self.width,     self.height))

        self.prev = None
        self.frame_counter = 0
        self.light_change_counter = 0
        self.persons = []
        self.current_person_id = 1
        self.counter = 0
        self.contours = []
        self.bgmask = None
        self.bgmodel = None
        self.lightChange = False

        print('''
Input:
    width : {} px
    height: {} px
    fps: {}
Output:
    width : {} px
    height: {} px
        '''.format(self.width, self.height, self.fps, self.out_width, self.out_height))

    def get_next_video_frame(self):
        with self._capture_lock:
            return self.cap.read()

    def create_background_model(self):
        print('create_background_model')
        _, self.bgmodel = self.get_next_video_frame()
        self.bgmask = 255 * np.ones((self.height, self.width), dtype=np.uint8)
        #if self.display_window:
            #cv2.imshow('video', self.bgmodel)
            #cv2.imshow('bgModel', self.bgmodel)

        #for _ in xrange(700):
        #    ret, frame = self.get_next_video_frame()
        #    continue;

        for _ in range(self.params['n_first_frames']):
            ret, frame = self.get_next_video_frame()
            self.bgmodel = cv2.addWeighted(self.bgmodel, 0.5, frame, 0.5, 0)
            #if self.display_window:
                #cv2.imshow('video', frame)
                #cv2.imshow('bgModel', self.bgmodel)

        ##frame_conv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ##_, _, Vf = cv2.split(frame_conv)
        ##self.prev = np.mean(Vf)

    def start(self):
        self.lightChange = False

        while(self.cap.isOpened()):
            ret, frame = self.get_next_video_frame()
            if ret == False:
                break
            self.process_frame(frame)
            self.wait_esc_key()

    def process_frame(self, frame):
        if self.bgmodel is None:
            self.create_background_model()

        if not self.lightChange:
            if self.frame_counter % self.params['update_bg_every_n_frames'] == 0:
                # Take invariant bgmodel ROIs (where person is standing)
                bg_const = cv2.bitwise_and(self.bgmodel, self.bgmodel, mask=self.bgmask)
                if self.display_window:
                    cv2.imshow('bgmask', self.bgmask)

                self.bgmodel = cv2.addWeighted(self.bgmodel, 0.89, frame, 0.11, 0) # Update bgmodel
                bgmask_inv = cv2.bitwise_not(self.bgmask) # Create inverse bgmask
                self.bgmodel = cv2.bitwise_and(self.bgmodel, self.bgmodel, mask=bgmask_inv) # Black out invariant bgmodel ROIs
                self.bgmodel = cv2.add(self.bgmodel, bg_const) # Add invariant bgmodel ROIs to the updated bgmodel
                self.bgmask = np.zeros((self.height, self.width), dtype=np.uint8) # Create new bgmask for the next bgmodel update
                #if self.display_window:
                    #cv2.imshow('bgModel', self.bgmodel)

        # self._light_change_solution_1(frame)

    #    if self.save_output:
    #        self.outStream2.write(frame) Write the frame

        self.frame_counter += 1
        if self.frame_counter % self.params['n_frames'] != 0:
            #if self.display_window:
                #cv2.imshow('video', frame)
            return
        elif self.frame_counter > 2e2:
            self.frame_counter = 0

        frame_orig = frame.copy()
        frame_conv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        bgmodel_conv = cv2.cvtColor(self.bgmodel, cv2.COLOR_BGR2HSV)

        _, _, Vf = cv2.split(frame_conv)
        _, _, Vb = cv2.split(bgmodel_conv)
        diff = cv2.absdiff(Vf, Vb)

    ##    avg = np.mean(Vf)
    ##    if abs(self.prev-avg) > self.params['light_threshold']:        # light change analysis
    ##        self.lightChange = True
    ##        self.frame_counter = 0
    ##        self.light_change_counter = 0
    ##    self.prev = avg

    # self._light_change_solution_2(frame)

    #    diff = cv2.erode(diff, self.kernel, iterations=1)
    #    diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, self.kernel)
    #    diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN,  self.kernel)

        ret, diff_thres = cv2.threshold(diff, self.params['threshold'], 255, cv2.THRESH_BINARY)
        diff_thres = cv2.morphologyEx(diff_thres, cv2.MORPH_OPEN,  self.kernelOp) #Opening (erode  ->  dilate) para quitar ruido.
        diff_thres = cv2.morphologyEx(diff_thres, cv2.MORPH_CLOSE, self.kernelCl) #Closing (dilate ->  erode)  para juntar regiones blancas.

        self.counter = 0
        self.contours, hierarchy = cv2.findContours(diff_thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]

        for cnt in self.contours:
#D            cv2.drawContours(frame, cnt, -1, (0,255,0), 3, 8) ### DRAW ###
            area = cv2.contourArea(cnt)
            if area > self.params['area_threshold']:
                #print("COUNT: " + str(self.counter))
                #################
                #   TRACKING    #
                #################
                M = cv2.moments(cnt)
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                bounding_rect = cv2.boundingRect(cnt)
                x,y,w,h = bounding_rect
                self.counter += 1

                new = True
                for p in self.persons:
                    if abs(cx-p.getX()) <= w and abs(cy-p.getY()) <= h:            # if abs(x-p.getX()) <= w and abs(y-p.getY()) <= h:
                        # el objeto esta cerca de uno que ya se detecto antes
                        new = False
                        #if p.i == 1:
                        #    print p.i, ": ", cx, " ", cy
                        #    print "X: ", p.getX(), " ", p.getY()
                        if cx == p.getX() and cy == p.getY():
                            p.age_one()
                            #print p.age
                        else:
                            p.age = 0
                        #if p.isStanding():
                            # Update bgmask with the ROI where person is standing
                            #print "STANDING"
                        self.bgmask[y:y+h, x:x+w] = 255 * np.ones((h, w), dtype=np.uint8)

                        p.updateCoords(cx,cy, bounding_rect)   #actualiza coordenadas en el objeto and resets age
                        break
                if new == True:
                    newp = Person.MyPerson(self.current_person_id, cx, cy, bounding_rect, self.params['max_person_age'])
                    self.persons.append(newp)
                    self.current_person_id += 1

                    self.mqtt_fn('Person detected')
                #################
                #   DIBUJOS     #
                #################
#D                cv2.circle(frame, (cx,cy), 5, (0,0,255), -1) ### DRAW ###
#D                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2) ### DRAW ###
#D                cv2.drawContours(frame, cnt, -1, (0,255,0), 3) ### DRAW ###

        #########################
        # DIBUJAR TRAYECTORIAS  #
        #########################
#D        for p in self.persons:
#D            if len(p.getTracks()) >= 2:
#D                pts = np.array(p.getTracks(), np.int32)
#D                pts = pts.reshape( (-1,1,2) )
#D                ### DRAW ### frame = cv2.polylines(frame, [pts], False, p.getRGB())
#_#            if p.getId() == 9:
#_#                print(str(p.getX()), ',', str(p.getY()))
#D            cv2.putText(frame, str(p.getId()), (p.getX(),p.getY()), self.font, 0.3, p.getRGB(), 1, cv2.LINE_AA) ### DRAW ###

#D        cv2.putText(frame, "COUNTER: " + str(self.counter),(int(0.8*self.width),30), self.font, 0.5, (0,255,0), 1, cv2.LINE_AA) ### DRAW ###

        out1 = frame.copy()
        if self.save_output:
            self.outStream2.write(frame) # Write the frame

        if self.save_output or self.display_window:
            frame_orig     = cv2.copyMakeBorder(frame_orig,   0,3,0,3, cv2.BORDER_CONSTANT, value=[0,255,0])
            frame          = cv2.copyMakeBorder(frame,        0,3,3,0, cv2.BORDER_CONSTANT, value=[0,255,0])
            bgmodel_border = cv2.copyMakeBorder(self.bgmodel, 3,0,0,3, cv2.BORDER_CONSTANT, value=[0,255,0])
            diff_thres     = cv2.copyMakeBorder(diff_thres,   3,0,3,0, cv2.BORDER_CONSTANT, value=[0,255,0])

            #cv2.putText(frame, "COUNTER: " + str(self.counter),(int(0.8*self.width),30), self.font, 0.5, (0,255,0), 1, cv2.LINE_AA)

            diff_thres = cv2.cvtColor(diff_thres, cv2.COLOR_GRAY2BGR)
            top    = np.concatenate((frame_orig, frame),          axis=1)
            bottom = np.concatenate((bgmodel_border, diff_thres), axis=1)
            whole  = np.concatenate((top, bottom),                axis=0)

            out2 = whole.copy()
            if self.save_output:
                self.outStream1.write(whole) # Write the frame

            whole = cv2.resize(whole, (int(whole.shape[1]/1.5), int(whole.shape[0]/1.5)), interpolation = cv2.INTER_AREA)
            if self.display_window:
                cv2.imshow('video2', whole)

        if self.out_width != self.width and self.out_height != self.height:
            out1 = cv2.resize(out1, (self.out_width, self.out_height), interpolation=cv2.INTER_AREA)
        return out1 #, out2

    def draw_overlay(self, frame):
        for cnt in self.contours:
            cv2.drawContours(frame, cnt, -1, (0,255,0), 3, 8)

        for p in self.persons:
            cx = p.getX()
            cy = p.getY()
            x,y,w,h = p.getBoundingRect()

            cv2.circle(frame, (cx,cy), 5, (0,0,255), -1)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            #cv2.drawContours(frame, cnt, -1, (0,255,0), 3)

            if len(p.getTracks()) >= 2:
                pts = np.array(p.getTracks(), np.int32)
                pts = pts.reshape( (-1,1,2) )
                cv2.polylines(frame, [pts], False, p.getRGB())
            cv2.putText(frame, str(p.getId()), (cx,cy), self.font, 0.3, p.getRGB(), 1, cv2.LINE_AA)

        cv2.putText(frame, "COUNTER: " + str(self.counter),(int(0.8*self.width),30), self.font, 0.5, (0,255,0), 1, cv2.LINE_AA)

    def _light_change_solution_1(self, frame):
        ''' Update background model more frequently '''
        if self.lightChange:
            if self.frame_counter % self.params['update_bg_every_n_frames']/5 == 0:
                self.bgmodel = cv2.addWeighted(self.bgmodel, 0.89, frame, 0.11, 0)
                #if self.display_window:
                    #cv2.imshow('bgModel', self.bgmodel)
                self.light_change_counter += 1
                if self.light_change_counter == 4*self.params['update_bg_every_n_frames']/5:        # for 5 updates, 5x more often
                    self.lightChange = False

    def _light_change_solution_2(self, frame):
        ''' Create background model from scratch '''
        if self.lightChange:
            ret, self.bgmodel = self.get_next_video_frame()
            for _ in xrange(self.params['n_first_frames']):
                ret, frame = self.get_next_video_frame()
                self.bgmodel = cv2.addWeighted(self.bgmodel, 0.5, frame, 0.5, 0)
                #if self.display_window:
                #cv2.imshow('video', frame)
                #cv2.imshow('bgModel', self.bgmodel)
            self.lightChange = False

    @staticmethod
    def wait_esc_key(delay=30):
        k = cv2.waitKey(delay) & 0xff
        return k == 27

    def cleanup(self):
        self.cap.release()
        if self.outStream1:
            self.outStream1.release()
        if self.outStream2:
            self.outStream2.release()
        cv2.destroyAllWindows()


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else 0
    pc = PersonCounter(source)
    pc.start()

if __name__ == '__main__':
    main()
