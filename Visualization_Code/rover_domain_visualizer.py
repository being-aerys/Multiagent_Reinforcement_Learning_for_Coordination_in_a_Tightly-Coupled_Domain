#!/usr/bin/env python3
#  This is a visualization tool designed to aid the rover domain imaging

# Created by Nick. Modified by Ashwin Vinoo
# Date: 2/7/2019

# import all the necessary modules
import numpy as np
import random
import pygame
import time
import math
import os

# ----------- Hyper Parameters -----------
# The main caption displayed over the pygame screen
primary_caption = 'Rover Domain Visualization'
# The text to display at the bottom of the screen
text_at_bottom = 'Simulation by Enna Sachdeva, Aashish Adhikari, Ashwin Vinoo'
# This is the font size of the text shown over the pygame display
pygame_font_size = 40
# This is the font shown over the pygame display
pygame_font = 'Times New Roman'

# The background image file name to use
background_image_location = 'background_9.png'
# The rover image file location to use
rover_image_location = 'rover_9.png'
# The green POI image file location
green_poi_image_location = 'poi_green_4.png'
# The red POI image file location
red_poi_image_location = 'poi_red_4.png'
# The observation ring image file location
observation_ring_image_location = 'observation_ring_8.png'
# This will allow you to disable the observation ring display
display_rings = True

# The ratio between the rover height and the the average of the window height and width
rover_image_scale = 0.07
# The ratio between the rover height and the the average of the window height and width
poi_image_scale = 0.04

# The width of the window to run the simulation
window_width = 1080
# The height of the window to run the simulation
window_height = 1080
# The displacement of the pygame window from the left border of the screen
window_width_offset = 78
# The displacement of the pygame window from the upper border of the screen
window_height_offset = 0

# The ratio between the rover trail circle radius and the average of the window height and width
circle_radius_ratio = 0.003
# The ratio between the rover trail line width and the average of the window height and width
line_width_ratio = 0.003
# -----------------------------------------

# The background image file location is appended with the folder locations
background_image_location = './display_images/background/' + background_image_location
# The rover image file location is appended with the folder locations
rover_image_location = './display_images/rovers/' + rover_image_location
# The green POI image file location is appended with the folder locations
green_poi_image_location = './display_images/poi/' + green_poi_image_location
# The green POI image file location is appended with the folder locations
red_poi_image_location = './display_images/poi/' + red_poi_image_location
# The green POI image file location is appended with the folder locations
observation_ring_image_location = './display_images/observation_ring/' + observation_ring_image_location


# The rover domain visualizer class is defined here
class RoverDomainVisualizer(object):

    # This function helps visualize the rover domain environment
    # resolution is the size of the pygame window
    # window offset allows us to move the pygame window to the right or downwards (makes space for the task bar access)
    def __init__(self, rover_count, grid_size, observation_radius, coupling_factor,
                 resolution=(window_width, window_height), window_offset=(window_width_offset, window_height_offset)):

        # The pygame window will be initialized with these displacements from the border of the screen
        os.environ['SDL_VIDEO_WINDOW_POS'] = str(window_offset[0]) + "," + str(window_offset[1])
        # Initialize all imported pygame modules
        pygame.init()
        # Load a new font from a given filename or a python file object. The size is the height of the font in pixels
        self.pygame_font = pygame.font.SysFont(pygame_font, pygame_font_size)
        # Sets the pygame display to cover the maximum space permitted as we are ignoring the resolution argument
        self.pygame_display = pygame.display.set_mode(resolution, pygame.NOFRAME)
        # Sets the caption to be displayed on the window
        pygame.display.set_caption(primary_caption)
        # Get the current pygame window information
        display_info = pygame.display.Info()
        # Obtains the pygame window width
        self.display_width = display_info.current_w
        # Obtains the pygame window height
        self.display_height = display_info.current_h
        # The grid width that is used in the rover domain
        self.grid_width = grid_size[0]
        # The grid height that is used in the rover domain
        self.grid_height = grid_size[1]
        # The coupling factor
        self.coupling_factor = coupling_factor

        # We create the text to display as a font object
        self.text_surface = self.pygame_font.render(text_at_bottom, True, (0, 0, 0))
        # We obtain the width of the text surface
        self.text_surface_width = self.text_surface.get_width()
        # We obtain the height of the text surface
        self.text_surface_height = self.text_surface.get_height()

        # Loads the background image
        self.background_image = pygame.image.load(background_image_location)
        # Loads the robot image
        self.rover_image = pygame.image.load(rover_image_location)
        # Loads the green flag image
        self.green_poi_image = pygame.image.load(green_poi_image_location)
        # Loads the red flag image
        self.red_poi_image = pygame.image.load(red_poi_image_location)
        # Loads the observation ring
        self.observation_ring_image = pygame.image.load(observation_ring_image_location)

        # Obtains the rover image width
        rover_image_width = self.rover_image.get_width()
        # Obtains the rover image height
        rover_image_height = self.rover_image.get_height()
        # Obtains the green poi image width
        green_poi_image_width = self.green_poi_image.get_width()
        # Obtains the green poi image height
        green_poi_image_height = self.green_poi_image.get_height()
        # Obtains the red poi image width
        red_poi_image_width = self.red_poi_image.get_width()
        # Obtains the red poi image height
        red_poi_image_height = self.red_poi_image.get_height()

        # We obtain the average of the display height and width
        average_display_dimension = (self.display_width + self.display_height)/2
        # We obtain the average of the rover height and width
        average_rover_dimension = (rover_image_width + rover_image_height)/2
        # We obtain the average of the green POI height and width
        average_green_poi_dimension = (green_poi_image_width + green_poi_image_height)/2
        # We obtain the average of the red POI height and width
        average_red_poi_dimension = (red_poi_image_width + red_poi_image_height)/2

        # The scaled width of the rover image
        self.rover_width = int((rover_image_scale*average_display_dimension) *
                               (rover_image_width/average_rover_dimension))
        # The scaled height of the rover image
        self.rover_height = int((rover_image_scale*average_display_dimension) *
                                (rover_image_height/average_rover_dimension))
        # The scaled width of the green poi image
        self.green_poi_width = int((poi_image_scale*average_display_dimension) *
                                   (green_poi_image_width/average_green_poi_dimension))
        # The scaled height of the rover image
        self.green_poi_height = int((poi_image_scale*average_display_dimension) *
                                    (green_poi_image_height/average_green_poi_dimension))
        # The scaled width of the green poi image
        self.red_poi_width = int((poi_image_scale*average_display_dimension) *
                                 (red_poi_image_width/average_red_poi_dimension))
        # The scaled height of the rover image
        self.red_poi_height = int((poi_image_scale*average_display_dimension) *
                                  (red_poi_image_height/average_red_poi_dimension))
        # The scaled rover trail circle radius
        self.circle_radius = int(circle_radius_ratio*average_display_dimension)
        # The scaled rover trail line width
        self.line_width = int(line_width_ratio*average_display_dimension)
        # The observation radius for the POI
        self.observation_radius = observation_radius
        # The scaled diameter of the observation ring
        self.ring_diameter = int(4*observation_radius*average_display_dimension/(self.grid_width + self.grid_height))

        # We obtain the scaled background image
        self.background_image = pygame.transform.scale(self.background_image, (self.display_width, self.display_height))
        # We obtain the scaled rover image
        self.rover_image = pygame.transform.scale(self.rover_image, (self.rover_width, self.rover_height))
        # We obtain the scaled green poi image
        self.green_poi_image = pygame.transform.scale(self.green_poi_image,
                                                      (self.green_poi_width, self.green_poi_height))
        # We obtain the scaled red poi image
        self.red_poi_image = pygame.transform.scale(self.red_poi_image, (self.red_poi_width, self.red_poi_height))
        # We obtain the scaled observation ring image
        self.observation_ring_image = pygame.transform.scale(self.observation_ring_image,
                                                             (self.ring_diameter, self.ring_diameter))

        # The effective display width used to prevent clipping of rover and POI images placed near the vertical edges
        self.effective_display_width = self.display_width - np.max([self.rover_width, self.green_poi_width,
                                                                    self.red_poi_width])
        # The effective display height used to prevent clipping of rover and POI images placed near the horizontal edges
        self.effective_display_height = self.display_height - np.max([self.rover_height, self.green_poi_height,
                                                                      self.red_poi_height])

        # We generate a list of RGB colors for marking the trail of each rover
        self.trail_color_array = self.generate_random_colors(rover_count)
        # This list holds lists (for each time step) containing the coordinates of each rover
        self.rover_trail_list = []
        # The POI status list that is updated each round
        self.poi_status_list = []

    # This function updates the pygame display with the passed parameters
    # rover_pos_list contains the rover coordinates as a list  eg: [(1,2),(2,3),(3,4)]
    # poi_pos_list contains the POI coordinates as a list  eg: [(1,2),(2,3),(3,4)]
    # poi_status_list contains a list of the observation status of the POIs eg: [True, False, True]
    # Reset is used if we want to clear the rover trails when we end the episode
    # Wait_time is the time_delay to pause between frames
    def update_visualizer(self, rover_pos_list, poi_pos_list, reset=False, wait_time=0.1):

        # We draw the background on the screen to erase everything previously drawn over the window
        self.pygame_display.blit(self.background_image, (0, 0))

        # We display the text on the bottom of the display screen
        self.pygame_display.blit(self.text_surface, ((self.display_width - self.text_surface_width) / 2,
                                                     self.display_height - 1.5 * self.text_surface_height))

        # We display the observation ring if True
        if display_rings:
            # We iterate through the poi list to draw the observation rings
            for i in range(len(poi_pos_list)):
                # We obtain the poi coordinate
                poi_coordinate = poi_pos_list[i]
                # The adjusted x coordinate for the poi
                adjusted_x_coordinate = int(self.effective_display_width * (poi_coordinate[0] / self.grid_width))
                # The adjusted y coordinate for the poi
                adjusted_y_coordinate = int(self.effective_display_height * (poi_coordinate[1] / self.grid_height))
                # We display the observation ring over the coordinate adjusted for the window size
                self.pygame_display.blit(self.observation_ring_image,
                                         (adjusted_x_coordinate-self.ring_diameter/2+self.green_poi_width/2,
                                          adjusted_y_coordinate-self.ring_diameter/2+self.green_poi_height/2))

        # We create an adjusted rover coordinate list to help form the trail paths
        adjusted_rover_pos_list = []
        # We iterate through the rover list
        for coordinate in rover_pos_list:
            # The adjusted x coordinate for the rover
            adjusted_x_coordinate = int(self.effective_display_width * (coordinate[0] / self.grid_width))
            # The adjusted y coordinate for the rover
            adjusted_y_coordinate = int(self.effective_display_height * (coordinate[1] / self.grid_height))
            # We appended the adjusted coordinates to the adjusted rover position list
            adjusted_rover_pos_list.append((adjusted_x_coordinate, adjusted_y_coordinate))

        # If the reset flag is True, we should reset the rover trail list
        if reset:
            # This list holds lists (for each time step) containing the coordinates of each rover
            self.rover_trail_list = []
            # We add the rover list for the current time step
            self.rover_trail_list.append(adjusted_rover_pos_list)
        else:
            # We append the rover position list to the rover trail list
            self.rover_trail_list.append(adjusted_rover_pos_list)

        # Iterating through the time steps recorded
        for time_step in range(len(self.rover_trail_list)):
            # Iterate through all the rovers in the rover trail list
            for rov_id in range(len(adjusted_rover_pos_list)):
                # The RGB color combination to use to represent the path
                rgb_color = self.trail_color_array[rov_id]
                # We obtain the adjusted rover coordinate at this time step
                adjusted_rover_pos = self.rover_trail_list[time_step][rov_id]
                # In case the time step is the latest one
                if time_step == len(self.rover_trail_list)-1:
                    # We display the rover image over the coordinate adjusted for the window size
                    self.pygame_display.blit(self.rover_image, adjusted_rover_pos)
                # In case the time step is an earlier one
                else:
                    # Draws the trail circles at that time step
                    pygame.draw.circle(self.pygame_display, rgb_color,
                                       (adjusted_rover_pos[0] + int(self.rover_width / 2),
                                        adjusted_rover_pos[1] + int(self.rover_height / 2)),
                                       self.circle_radius)

                    # We obtain the adjusted rover coordinate at the next time step
                    next_step_adjusted_rover_pos = self.rover_trail_list[time_step+1][rov_id]
                    # Draws the trail line
                    pygame.draw.line(self.pygame_display, rgb_color,
                                     (adjusted_rover_pos[0] + int(self.rover_width / 2),
                                      adjusted_rover_pos[1] + int(self.rover_height / 2)),
                                     (next_step_adjusted_rover_pos[0] + int(self.rover_width / 2),
                                      next_step_adjusted_rover_pos[1] + int(self.rover_height / 2)),
                                     self.line_width)

        # We update the POI statuses
        self.update_poi_status(rover_pos_list, poi_pos_list, reset)

        # We iterate through the poi list
        for i in range(len(poi_pos_list)):
            # We obtain the poi coordinate
            poi_coordinate = poi_pos_list[i]
            # We obtain the poi status
            poi_status = self.poi_status_list[i]
            # The adjusted x coordinate for the poi
            adjusted_x_coordinate = int(self.effective_display_width * (poi_coordinate[0] / self.grid_width))
            # The adjusted y coordinate for the poi
            adjusted_y_coordinate = int(self.effective_display_height * (poi_coordinate[1] / self.grid_height))
            # If the poi is observed
            if poi_status:
                # We display the green poi image over the coordinate adjusted for the window size
                self.pygame_display.blit(self.green_poi_image, (adjusted_x_coordinate, adjusted_y_coordinate))
            # If the poi is not observed
            else:
                # We display the green poi image over the coordinate adjusted for the window size
                self.pygame_display.blit(self.red_poi_image, (adjusted_x_coordinate, adjusted_y_coordinate))

        # Update portions of the screen for software displays
        pygame.display.update()
        # We sleep for a small period between frames. We make sure not to go beyond 30fps
        time.sleep(max(wait_time, 0.0333))

    # This function updates the POI status
    def update_poi_status(self, rover_pos_list, poi_pos_list, reset=False):
        # if the POI status is empty or we want to reset the episode
        if not self.poi_status_list or reset:
            # We use a POI status list of all False
            self.poi_status_list = [False for _ in range(len(poi_pos_list))]
        # We iterate through the poi list
        for i in range(len(poi_pos_list)):
            # We can skip this round if the POI status is already True
            if self.poi_status_list[i]:
                # We skip this round
                continue
            # We obtain the poi coordinate
            poi_coordinate = poi_pos_list[i]
            # This represents the number of rovers that have coupled to the current POI
            coupled_rovers = 0
            # We iterate through the rover list
            for coordinate in rover_pos_list:
                # We check the euclidean distance between current POI and rover
                euclidean = math.sqrt((poi_coordinate[0]-coordinate[0])**2 + (poi_coordinate[1]-coordinate[1])**2)
                # In case the euclidean distance is less than or equal to the POI observation radius
                if euclidean <= self.observation_radius:
                    # We increment the count of the number of rovers coupled to the POI at current time step
                    coupled_rovers += 1
                # We check if the number of coupled rovers is equal to the coupling factor
                if coupled_rovers == self.coupling_factor:
                    # We mark this POI status as True
                    self.poi_status_list[i] = True
                    # We break out of the rover for loop
                    break

    @staticmethod
    # This function generates the specified number of random colors as an list
    def generate_random_colors(number_of_colors, default=((255, 0, 0), (0, 200, 0), (0, 0, 255), (128, 0, 128))):
        # The list of colors is initialized to an empty list
        color_list = []
        # We iterate through the for loop 'number_of_colors' times
        for i in range(number_of_colors):
            # In case the iteration is less than the length of the default color list
            if i <= len(default):
                # We append the default colors
                color_list.append(default[i])
            else:
                # We append a random RGB vector eg:[1, 200, 135] to the color list
                color_list.append(list(np.random.choice(range(256), size=3)))
        # Returns the color list
        return color_list

# -------------------------------- CODE FOR TESTING ONLY --------------------------------

# If this file is the main one called for execution
if __name__ == "__main__":

    # The number of rovers in the example
    rover_count = 4
    # The number of POIs
    poi_count = 10
    # The number of rounds to run
    rounds_to_run = 1000
    # The reset interval after which we remove all the trails
    reset_interval = 100
    # The step size we move the rovers each time step
    step_size = 3
    # The grid size
    grid_size = (30, 30)
    # observation radius of the POI
    observation_radius = 5
    # The coupling factor to observe a POI
    coupling_factor = 2
    # The probability of observing a POI each round
    poi_prob = 0.05
    # The wait time between frames
    waiting_time = 0.1

    # We initialize the visualization object with the number of rovers and the grid size (width x height)
    visualizer = RoverDomainVisualizer(rover_count, grid_size, observation_radius, coupling_factor)

    # Initializing the rover coordinate list
    rover_coordinate_list = []
    # Initializing the poi coordinate list
    poi_coordinate_list = []
    # Generating a POI status list
    poi_status_list = []
    # We iterate through the number of rounds to run
    for i in range(rounds_to_run):
        # We check if it is the round to remove the trails
        if i % reset_interval == 0:
            # We decide to reset
            reset = True
            # Initializing the rover coordinate list
            rover_coordinate_list = [(grid_size[0] * random.random(), grid_size[1] * random.random()) for _ in range(rover_count)]
            # Initializing the poi coordinate list
            poi_coordinate_list = [(grid_size[0] * random.random(), grid_size[1] * random.random()) for _ in range(poi_count)]
        # We don't need to reset
        else:
            # We don't reset
            reset = False

        # We update the visualizer
        visualizer.update_visualizer(rover_coordinate_list, poi_coordinate_list, reset, waiting_time)

        # Here we move the rovers for the next round
        for j in range(rover_count):
            # The rover x coordinate
            rover_x = rover_coordinate_list[j][0]
            # The rover y coordinate
            rover_y = rover_coordinate_list[j][1]
            # The rover randomly moves along x-axis by step width
            rover_x += step_size*random.random() - step_size/2
            # The rover randomly moves along y-axis by step width
            rover_y += step_size*random.random() - step_size/2
            # We limit the allowable x-axis range
            rover_x = min(grid_size[0], max(0, rover_x))
            # We limit the allowable y-axis range
            rover_y = min(grid_size[1], max(0, rover_y))
            # We update the new rover x and y coordinates
            rover_coordinate_list[j] = (rover_x, rover_y)
