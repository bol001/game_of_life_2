import matplotlib.pyplot as plt
import numpy as np
from random import randrange
from matplotlib import colors
import matplotlib.patches as patches
import time

# Set how many generations per second and calculate the duration of each
# generation in seconds
generations_per_second = 5
generation_duration = 1 / generations_per_second

# Define size of overall grid and create initial emtpy np array
array_dimension_length = 1400
grid_array = np.full(
    (array_dimension_length, array_dimension_length),
    False
)

# Set random state of grid. First determine how many cells will have their state
# set to True.
grid_element_count = grid_array.size
number_of_elements_to_set = randrange(grid_element_count) + 1

# Randomly select the cells to update and update them. Iterate over the range
# of the number of elements we want to update.
for i in range(number_of_elements_to_set):

    # Randomly select an element from the grid
    random_x = randrange(array_dimension_length)
    random_y = randrange(array_dimension_length)

    # Update the selected element
    grid_array[random_x, random_y] = True

# Use a static grid state for debugging
# grid_array[0, 0] = True
# grid_array[0, 1] = True
# grid_array[0, 2] = True

# Print initial grid array states for debugging
# np.set_printoptions(threshold=sys.maxsize)
# print(grid_array)

# Set plot dimensions (relative to screen size)
plot_dimension_x = 14
plot_dimension_y = 8

# Instantiate matplotlib figure and axes
fig, ax = plt.subplots(
    1, 2,  # rows, columns for subplots
    figsize=(plot_dimension_x, plot_dimension_y),
    gridspec_kw={'height_ratios': [0.5]}  # set ratio of plots
)

# Set width of border area around grid
fig.subplots_adjust(left=0.025, right=0.975, top=0.975, bottom=0.025)

# Create the matplotlib axes (left most is main grid, rightmost is most active
# tile). Because we have not determined the most active tile yet, just set the
# right most axes to display the whole grid.
ax[0].matshow(grid_array, cmap='binary', vmin=0, vmax=1)
ax[1].matshow(grid_array, cmap='binary', vmin=0, vmax=1)

# Set border area to grey
fig.patch.set_facecolor('gray')

# Don't show the axis labels
plt.axis('off')

# Show the starting matplotlib plot. Set block to False so the running of the
# plot can be interrupted to update it.
plt.show(block=False)


def get_cell_next_generation_state(value_pair):

    # Get the grid state value and the live neighbor count value
    grid_state_value = value_pair[0]
    live_neighbor_count_value = value_pair[1]

    # Determine the next generation state for this cell
    if not grid_state_value:
        # Case of this cell is dead

        if live_neighbor_count_value == 3:
            # Case of this cell is dead and neighbors are such
            # that cell will come alive

            cell_next_generation_state = True

        else:
            # Case of this cell is dead and there are no
            # changes to in the next generation

            cell_next_generation_state = grid_state_value

    else:
        # Case of this cell is alive

        if (
                live_neighbor_count_value in (0, 1)
                or
                live_neighbor_count_value >= 4
        ):
            # Case of this cell is alive and neighbors are
            # such that cell will die.

            cell_next_generation_state = False

        elif live_neighbor_count_value in (2, 3):
            # Case of this cell is alive and neighbors are
            # such that this cell will stay alive

            cell_next_generation_state = True

        else:
            # Case of this cell is alive and there are no
            # changes to it in the next generation
            cell_next_generation_state = grid_state_value

    return cell_next_generation_state


# Run a loop to process each generation of the grid. Pause each time through
# for the duration of a generation. Also use a counter so we can update the
# number of generations in the plot.
counter = 2
plt.pause(generation_duration)
while True:

    generation_start = time.time()

    # Create additional versions of the current array which are shifted
    # and which represent the states of cell neighbors.

    grid_array_shifted_top_neighbor = np.vstack(
        [
            # New row of False values
            np.full(array_dimension_length, False),
            # Original grid
            grid_array
        ]
    )
    # Delete the last row
    grid_array_shifted_top_neighbor = (
        np.delete(grid_array_shifted_top_neighbor, -1, axis=0)
    )

    grid_array_shifted_top_left_neighbor = np.hstack(
        [
            # New column of False values added to left
            np.full((array_dimension_length + 1, 1), False),

            # On right:
            np.vstack(
                [
                    # New row of False values added to top
                    np.full(array_dimension_length, False),
                    # Original grid
                    grid_array
                ]
            )
        ]
    )
    # Delete last row and last column
    grid_array_shifted_top_left_neighbor = (
        np.delete(grid_array_shifted_top_left_neighbor, -1, axis=0)
    )
    grid_array_shifted_top_left_neighbor = (
        np.delete(grid_array_shifted_top_left_neighbor, -1, axis=1)
    )

    grid_array_shifted_top_right_neighbor = np.delete(
        # Start with:
        np.vstack(
            [
                # New row of False values added to top
                np.full(array_dimension_length, False),
                # Original grid
                grid_array
            ]
        ),
        # Delete first column
        0,
        axis=1
    )
    # Delete bottom row and add column on right
    grid_array_shifted_top_right_neighbor = (
        np.delete(grid_array_shifted_top_right_neighbor, -1, axis=0)
    )
    grid_array_shifted_top_right_neighbor = np.hstack(
        [
            # Original grid
            grid_array_shifted_top_right_neighbor,
            # New column of False values added to right
            np.full((array_dimension_length, 1), False)
        ]
    )

    grid_array_shifted_left_neighbor = np.hstack(

        [
            # New column of False values added to left
            np.full((array_dimension_length, 1), False),

            # Original grid
            grid_array
        ]
    )
    # Delete last column
    grid_array_shifted_left_neighbor = (
        np.delete(grid_array_shifted_left_neighbor, -1, axis=1)
    )

    grid_array_shifted_right_neighbor = np.delete(
        # Original grid
        grid_array,
        # Delete first column
        0,
        axis=1
    )
    # Add column of False values to right
    grid_array_shifted_right_neighbor = np.hstack(
        [
            # Original grid
            grid_array_shifted_right_neighbor,
            # New column of False values added to right
            np.full((array_dimension_length, 1), False)
        ]
    )

    grid_array_shifted_bottom_neighbor = np.delete(
        grid_array,
        # Delete first row
        0,
        axis=0
    )
    # Add row of False values to bottom
    grid_array_shifted_bottom_neighbor = np.vstack(
        [
            # Original grid
            grid_array_shifted_bottom_neighbor,
            # New row of False values added to bottom
            np.full(array_dimension_length, False)
        ]
    )

    grid_array_shifted_bottom_left_neighbor = np.hstack(
        [
            # New column of False values added to left
            np.full((array_dimension_length - 1, 1), False),

            # On right
            np.delete(
                grid_array,
                # Delete first row
                0,
                axis=0
            )
        ]
    )
    # Delete last column and add row to bottom
    grid_array_shifted_bottom_left_neighbor = (
        np.delete(grid_array_shifted_bottom_left_neighbor, -1, axis=1)
    )
    grid_array_shifted_bottom_left_neighbor = np.vstack(
        [
            # Original grid
            grid_array_shifted_bottom_left_neighbor,
            # New row of False values added to bottom
            np.full(array_dimension_length, False)
        ]
    )

    grid_array_shifted_bottom_right_neighbor = np.delete(
        # Start with
        np.delete(
            grid_array,
            # Delete first row
            0,
            axis=0
        ),
        # Delete first column
        0,
        axis=1
    )
    # Add row to bottom and column to right
    grid_array_shifted_bottom_right_neighbor = np.vstack(
        [
            # Original grid
            grid_array_shifted_bottom_right_neighbor,
            # New row of False values added to bottom
            np.full(array_dimension_length - 1, False),
        ]
    )
    grid_array_shifted_bottom_right_neighbor = np.hstack(
        [
            # Original grid
            grid_array_shifted_bottom_right_neighbor,
            # New column of False values added to right
            np.full((array_dimension_length, 1), False)
        ]
    )

    # Add the all the shifted arrays together to create a single array that
    # represents the number of live neighbors for each cell. Note that we
    # use the .astype() method to convert the boolean values to integers.
    grid_array_live_neighbor_counts = sum(
        [
            grid_array_shifted_top_neighbor.astype(int),
            grid_array_shifted_top_left_neighbor.astype(int),
            grid_array_shifted_top_right_neighbor.astype(int),
            grid_array_shifted_left_neighbor.astype(int),
            grid_array_shifted_right_neighbor.astype(int),
            grid_array_shifted_bottom_neighbor.astype(int),
            grid_array_shifted_bottom_left_neighbor.astype(int),
            grid_array_shifted_bottom_right_neighbor.astype(int)
        ]
    )

    # Now that we have one array that represents the current grid state
    # and another array that represents the number of live neighbors,
    # combine both these 2D arrays into a single 3D array. Note that
    # when we combine these arrays, because the entire array must
    # contain the same type of values, the boolean values in the first
    # array will be converted to integers (0 for False and 1 for True).
    grid_array_and_live_neighbor_counts = np.array(
        (
            grid_array,
            grid_array_live_neighbor_counts
        )
    )

    # In this 3D array, the first axis (axis 0) has two elements.
    # The first element is the array with the current grid state
    # (with rows [axis 1] and columns [axis 2]). The second
    # element is the array with the number of live neighbors
    # (also with rows [axis 1] and columns [axis 2]). On this 3D
    # array, we will apply a function along the 0 axis that will
    # iterate over the combination of pairs of values from both
    # the grid state array and the live neighbor count array.
    # Each pair of values that will be iterated over will be like
    # this: (grid state value, live neighbor count value). Above
    # we defined the function that will be applied along the 0 axis.
    # This function takes a pair of values and returns the new state
    # for the cell in the next generation.
    next_generation_grid_states = np.apply_along_axis(
        get_cell_next_generation_state,
        0,
        grid_array_and_live_neighbor_counts
    )

    # Now that we have all the states, update the main grid with the next
    # generation of states.
    grid_array = next_generation_grid_states

    generation_end = time.time()
    generation_duration = generation_end - generation_start

    most_active_portion_start = time.time()

    # Create an array for the right-hand grid that is 1/n of a dimension of the
    # main grid. Start by defining the ration of this small grid to the larger
    # grid and then calculate the dimension of the small grid.
    small_grid_ratio = 5
    small_grid_dimension = int(array_dimension_length / small_grid_ratio)

    # Iterate over the main grid to determine which small grid slice is the most
    # active. First create an empty list of the differences for each small box
    # (from the last generation to this generation) so we can add to this list
    # as we iterate over each small box.
    small_grid_differences = []

    # We will eventually select one of the small grids as the one to be plotted,
    # so define this as an empty array for now.
    small_grid_array = np.array([])

    # We will eventually record one the starting positions of each axis of the
    # small grid within the larger grid. Set these values to None for now.
    small_box_row_start_in_main_grid = None
    small_box_col_start_in_main_grid = None

    # Start iteration. Iterate over rows. Within each row iterate over columns.
    for r in range(small_grid_ratio):

        for c in range(small_grid_ratio):

            # Define where this small grid appears within the larger grid.
            small_box_row_start = r * small_grid_dimension
            small_box_row_end = small_box_row_start + small_grid_dimension
            small_box_col_start = c * small_grid_dimension
            small_box_col_end = small_box_col_start + small_grid_dimension

            # Get the *new* small array for this box
            this_new_small_grid_array = next_generation_grid_states[
                                        small_box_row_start: (
                                                small_box_row_end + 1
                                        ),
                                        small_box_col_start: (
                                                small_box_col_end + 1
                                        )
                                        ]

            # Get the *last* small array for this box.
            this_last_small_grid_array = grid_array[
                                         small_box_row_start: (
                                                 small_box_row_end + 1
                                         ),
                                         small_box_col_start: (
                                                 small_box_col_end + 1
                                         )
                                         ]

            # Set size of sample (will be 10 percent of the small grid)
            sample_size = round(this_new_small_grid_array.size / 10)

            # Create a new random generator
            random_generator = np.random.default_rng()

            # Get a random set of indices
            selected_indices = random_generator.choice(
                this_new_small_grid_array.size,
                size=sample_size,
                replace=False
            )

            # Create the sample arrays from both arrays we want to compare
            this_new_small_grid_array_sample = (
                this_new_small_grid_array.flatten()[selected_indices]
            )

            this_last_small_grid_array_sample = (
                this_new_small_grid_array.flatten()[selected_indices]
            )

            # Calculate the difference between the samples from the two arrays
            # (initial state of the small array and the next state of the small
            # array).
            small_grid_diff_array = np.logical_and(
                this_new_small_grid_array_sample,
                this_last_small_grid_array_sample
            )
            small_grid_diff_sum = small_grid_diff_array.sum()

            # Add the difference for this small box to the list of differences.
            small_grid_differences.append(small_grid_diff_sum)

            # If the differences for this small box is the max of all the
            # differences for all the small boxes so far, set this small box
            # array as the one we will use in the plot. Also record the upper
            # left corner positions for this small grid within the larger grid
            # so that we can later highlight the given smaller grid within the
            # larger grid.
            if small_grid_diff_sum == max(small_grid_differences):

                # Define the small grid array
                small_grid_array = this_new_small_grid_array

                # Record the corner positions for each dimension of this small
                # grid
                small_box_row_start_in_main_grid = small_box_row_start
                small_box_col_start_in_main_grid = small_box_col_start

    most_active_portion_end = time.time()
    most_active_portion_duration = most_active_portion_end - most_active_portion_start

    # Update the generation counter
    counter = counter + 1

    # Clear the axes
    ax[0].clear()
    ax[1].clear()

    # Hide/re-hide tick marks and their labels
    ax[0].set_yticklabels([])
    ax[1].set_yticklabels([])
    ax[0].set_xticklabels([])
    ax[1].set_xticklabels([])
    ax[0].tick_params(axis='both', which='both', length=0)
    ax[1].tick_params(axis='both', which='both', length=0)

    # Update title (really the x axis label) of left hand plot so we can see
    # number of generations
    ax[0].set_xlabel(
        f'cells: {array_dimension_length ** 2:,.0f} | '
        f'max gens/s: {generations_per_second} | '
        f'gens: {counter:,.0f} | '
        f'gen secs: {generation_duration:,.1f} | '
        f'most active area detection secs: {most_active_portion_duration:,.2f}'
    )

    # Update the tile (really the x axis label) of the right hand plot to
    # indicate that it shows the most active part of the main grid.
    ax[1].set_xlabel('Most Active Portion of Overall Grid')

    # Update the axes to show the main grid and most active portion of the main
    # grid respectively.
    ax[0].matshow(grid_array, cmap='binary', vmin=0, vmax=1)
    small_grid_cmap = colors.ListedColormap(['white', 'red'])
    ax[1].matshow(small_grid_array, cmap=small_grid_cmap, vmin=0, vmax=1)

    # Draw a box around the portion of the main grid that represents the most
    # active section that is shown on the right.
    small_grid_box = patches.Rectangle(
        # Upper left-hand coordinate of the red box
        xy=(small_box_col_start_in_main_grid, small_box_row_start_in_main_grid),
        width=small_grid_dimension,
        height=small_grid_dimension,
        linewidth=1,
        edgecolor='red',
        facecolor='none'
    )
    ax[0].add_patch(small_grid_box)

    # Pause the plot so we can see its state for the duration of one generation.
    plt.pause(generation_duration)
