#!/usr/bin/env python3
import pygame
import random
import imageio
import sys

from utils.debug import debug_patches_per_frame
from core.property import Pos_x, Pos_y, Shape_x, Shape_y
# ========================

# ---------- Setup Pygame -----------
pygame.init()

# Choose a scale factor so that one “game unit” corresponds to, say, 30 pixels.
#scale = 30
# Based on our data (x roughly 0..22 and y roughly 0..12) we set the window size:

max_x = 0
max_y = 0
for frame in debug_patches_per_frame:
    for p in frame:
        if p.properties[Pos_x] + p.properties[Shape_x] > max_x: max_x = p.properties[Pos_x] + p.properties[Shape_x]
        if p.properties[Pos_y] + p.properties[Shape_y] > max_y: max_y = p.properties[Pos_y] + p.properties[Shape_y]

scale_x = 720 // max_x
scale_y = 360 // max_y

screen_width = (((max_x + 1) * scale_x) // 16) * 16
screen_height = ((((max_y + 1) * scale_y)) // 16) * 16

#screen_width = 24 * scale
#screen_height = 15.5 * scale
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Video from Patches")

# A dictionary to assign a unique (random) color to each patch name.
patch_colors = {}
def get_color(name):
    if name not in patch_colors:
        # Generate a random bright color.
        patch_colors[name] = (random.randint(50,255), random.randint(50,255), random.randint(50,255))
    return patch_colors[name]

# List to store frame images (for video saving)
frame_images = []

clock = pygame.time.Clock()
fps = 1

# Loop over each frame in our data structure.
for frame in debug_patches_per_frame:
    # Check for quit events so the user can close the window.
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Fill background with white.
    screen.fill((255, 255, 255))
    
    # For each patch in the current frame, compute its drawing rectangle
    # and draw it using its unique color.
    for patch in frame:
        props = patch.properties
        pos_x = props[Pos_x]
        pos_y = props[Pos_y]
        shape_x = props[Shape_x]
        shape_y = props[Shape_y]
        color = get_color(patch.description)
        
        # Compute the top-left corner and size in pixels.
        left = int((pos_x - shape_x) * scale_x)
        top = int((pos_y - shape_y) * scale_y)
        rect_width = int(2 * (shape_x + 0.5) * scale_x)
        rect_height = int(2 * (shape_y + 0.5) * scale_y)
        
        if rect_width == 0 and rect_height == 0:
            # Draw a small circle for a point (e.g. a ball)
            center = (int(pos_x * scale_x), int(pos_y * scale_y))
            pygame.draw.circle(screen, color, center, 5)
        elif rect_width == 0:
            # Vertical line
            start_pos = (int(pos_x * scale_x), int((pos_y - shape_y) * scale_y))
            end_pos   = (int(pos_x * scale_x), int((pos_y + shape_y) * scale_y))
            pygame.draw.line(screen, color, start_pos, end_pos, 3)
        elif rect_height == 0:
            # Horizontal line
            start_pos = (int((pos_x - shape_x) * scale_x), int(pos_y * scale_y))
            end_pos   = (int((pos_x + shape_x) * scale_x), int(pos_y * scale_y))
            pygame.draw.line(screen, color, start_pos, end_pos, 3)
        else:
            # Draw a filled rectangle.
            pygame.draw.rect(screen, color, (left, top, rect_width, rect_height))
    
    # Update the display.
    pygame.display.flip()
    
    # Capture the current frame (convert from (width,height,3) to (height,width,3))
    image = pygame.surfarray.array3d(screen).swapaxes(0,1)
    frame_images.append(image)
    
    # Wait until next frame (aiming at fps frames per second)
    clock.tick(fps)

# Save the collected frames as a video file.
output_filename = "output_video.mp4"
imageio.mimsave(output_filename, frame_images, fps=fps)
print(f"Video saved as {output_filename}")

# Wait for the user to close the window.
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

pygame.quit()
