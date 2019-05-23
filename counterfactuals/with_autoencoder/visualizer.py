#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pygame
import numpy as np
import time
import math
from D_VAE_Parameters import Parameters as p

pygame.font.init()  # you have to call this at the start, if you want to use this module
myfont = pygame.font.SysFont('Comic Sans MS', 30)


def draw(display, obj, x, y):
    display.blit(obj, (x, y))  # Correct for center of mass shift


def generate_color_array(num_colors): #generates num random colors
    color_arr = []
    
    for i in range(num_colors):
        color_arr.append(list(np.random.choice(range(256), size=3)))
    
    return color_arr


def visualize(rd, episode_reward):
    scale_factor = 25  # Scaling factor for images
    width = 32  # robot icon widths
    x_map = p.dim_x + 10  # Slightly larger so POI are not cut off
    y_map = p.dim_y + 10
    image_adjust = 100  # Adjusts the image so that everything is centered
    pygame.init()
    game_display = pygame.display.set_mode((x_map*scale_factor, y_map*scale_factor))
    pygame.display.set_caption('Rover Domain')
    robot_image = pygame.image.load('./robot.png')
    robot_image = pygame.transform.scale(robot_image, (64,  64))

    background = pygame.image.load('./background.png')
    background = pygame.transform.scale(background, (1536, 1536))

    greenflag = pygame.image.load('./greenflag.png')
    greenflag = pygame.transform.scale(greenflag, (64, 64))

    redflag = pygame.image.load('./redflag.png')
    redflag = pygame.transform.scale(redflag, (64, 64))

    color_array = generate_color_array(p.num_rovers * p.num_types)
    pygame.font.init() 
    myfont = pygame.font.SysFont('Comic Sans MS', 30)
    poi_status = [False for _ in range(p.num_pois)]
    
    for tstep in range(p.num_timesteps):
        draw(game_display, background, image_adjust, image_adjust)
        for poi_id in range(p.num_pois):  # Draw POI and POI values
            poi_x = int(rd.poi_pos[poi_id][0] * scale_factor-width) + image_adjust
            poi_y = int(rd.poi_pos[poi_id][1] * scale_factor-width) + image_adjust

            types_in_range = []; observer_count = 0
            for rover_id in range(p.num_types*p.num_rovers):
                x_dist = rd.poi_pos[poi_id][0] - rd.rover_path[rover_id][tstep][0]
                y_dist = rd.poi_pos[poi_id][1] - rd.rover_path[rover_id][tstep][1]
                #rover_type = rd.rover_path[rover_id][tstep][2]
                dist = math.sqrt((x_dist**2) + (y_dist**2))


                '''
                if p.team_types == "heterogeneous" and dist <= p.activation_dist:
                    #types_in_range.append(rover_type)
                elif p.team_types == "homogeneous" and dist <= p.activation_dist:
                    observer_count += 1
                '''

                if p.team_types == "homogeneous" and dist <= p.activation_dist:
                    observer_count += 1
                #if p.team_types == "heterogeneous":
            #    for t in range(p.num_types):
            #        if t in types_in_range:
            #            observer_count += 1

            if observer_count >= p.coupling:
                poi_status[poi_id] = True
            if poi_status[poi_id]:
                draw(game_display, greenflag, poi_x, poi_y)  # POI observed
            else:
                draw(game_display, redflag, poi_x, poi_y)  # POI not observed
            textsurface = myfont.render(str(rd.poi_value[poi_id]), False, (0, 0, 0))
            target_x = int(rd.poi_pos[poi_id][0]*scale_factor-scale_factor/3) + image_adjust
            target_y = int(rd.poi_pos[poi_id][1]*scale_factor-width) + image_adjust
            draw(game_display, textsurface, target_x, target_y)

        for rov_id in range(p.num_rovers * p.num_types):  # Draw all rovers and their trajectories
            rover_x = int(rd.rover_path[rov_id][tstep][0]*scale_factor) + image_adjust
            rover_y = int(rd.rover_path[rov_id][tstep][1]*scale_factor) + image_adjust
            draw(game_display, robot_image, rover_x, rover_y)

            if tstep != 0:  # start drawing trails from timestep 1.
                for timestep in range(1, tstep):  # draw the trajectory lines
                    line_color = tuple(color_array[rov_id])
                    start_x = int(rd.rover_path[rov_id][(timestep-1)][0]*scale_factor) + width/2 + image_adjust
                    start_y = int(rd.rover_path[rov_id][(timestep-1)][1]*scale_factor) + width/2 + image_adjust
                    end_x = int(rd.rover_path[rov_id][timestep][0]*scale_factor) + width/2 + image_adjust
                    end_y = int(rd.rover_path[rov_id][timestep][1]*scale_factor) + width/2 + image_adjust
                    line_width = 3
                    pygame.draw.line(game_display, line_color, (start_x, start_y), (end_x, end_y), line_width)
                    origin_x = int(rd.rover_path[rov_id][timestep][0]*scale_factor) + int(width/2) + image_adjust
                    origin_y = int(rd.rover_path[rov_id][timestep][1]*scale_factor) + int(width/2) + image_adjust
                    circle_rad = 3
                    pygame.draw.circle(game_display, line_color, (origin_x, origin_y), circle_rad)
        
        pygame.display.update()
        time.sleep(0.1)
        
    scoresurface = myfont.render('The system reward obtained is ' + str(round(episode_reward, 2)), False, (0, 0, 0))
    draw(game_display, scoresurface, x_map*scale_factor-500, 20)
    pygame.display.update()

    running = True  # Keeps visualizer from closing until you 'X' out of window
    pygame.quit()
    while False:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
