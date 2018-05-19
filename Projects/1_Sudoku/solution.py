
from utils import *
import itertools


row_units = [cross(r, cols) for r in rows]
column_units = [cross(rows, c) for c in cols]
square_units = [cross(rs, cs) for rs in ('ABC','DEF','GHI') for cs in ('123','456','789')]
unitlist = row_units + column_units + square_units

d1 = [''.join(ln) for ln in list(zip('ABCDEFGHI','123456789'))]
d2 = [''.join(ln) for ln in list(zip('ABCDEFGHI','123456789'[::-1]))]
unitlist.append(d1)
unitlist.append(d2)

# Must be called after all units (including diagonals) are added to the unitlist
units = extract_units(unitlist, boxes)
peers = extract_peers(units, boxes)


def naked_twins(values):
    """Eliminate values using the naked twins strategy.

    The naked twins strategy says that if you have two or more unallocated boxes
    in a unit and there are only two digits that can go in those two boxes, then
    those two digits can be eliminated from the possible assignments of all other
    boxes in the same unit.

    Parameters
    ----------
    values(dict)
        a dictionary of the form {'box_name': '123456789', ...}

    Returns
    -------
    dict
        The values dictionary with the naked twins eliminated from peers

    Notes
    -----
    Your solution can either process all pairs of naked twins from the input once,
    or it can continue processing pairs of naked twins until there are no such
    pairs remaining -- the project assistant test suite will accept either
    convention. However, it will not accept code that does not process all pairs
    of naked twins from the original input. (For example, if you start processing
    pairs of twins and eliminate another pair of twins before the second pair
    is processed then your code will fail the PA test suite.)

    The first convention is preferred for consistency with the other strategies,
    and because it is simpler (since the reduce_puzzle function already calls this
    strategy repeatedly).
    """

    def eliminate_non_twins(values, unit, twin):
        not_twins = list(set(unit) - set(twin[0]))
        for box in not_twins:
            values[box] = values[box].translate(str.maketrans('', '', twin[1]))
        return values

    def eliminate_unit(values, unit):
        l2_boxes = [box for box in unit if len(values[box]) == 2]
        pairs = list(itertools.combinations(l2_boxes, 2))
        twins = [((b1, b2), values[b1]) for b1, b2 in pairs if values[b1] == values[b2]]
        for twin in twins:
            values = eliminate_non_twins(values, unit, twin)
        return values

    for unit in unitlist:
        values = eliminate_unit(values, unit)
    return values


def eliminate(values):
    """Apply the eliminate strategy to a Sudoku puzzle

    The eliminate strategy says that if a box has a value assigned, then none
    of the peers of that box can have the same value.

    Parameters
    ----------
    values(dict)
        a dictionary of the form {'box_name': '123456789', ...}

    Returns
    -------
    dict
        The values dictionary with the assigned values eliminated from peers
    """
    new_values = values.copy()
    for key, value in values.items():
        if len(value) == 1:
            for peer in peers[key]:
                new_values[peer] = new_values[peer].replace(value[0], '')
    return new_values


def only_choice(values):
    """Apply the only choice strategy to a Sudoku puzzle

    The only choice strategy says that if only one box in a unit allows a certain
    digit, then that box must be assigned that digit.

    Parameters
    ----------
    values(dict)
        a dictionary of the form {'box_name': '123456789', ...}

    Returns
    -------
    dict
        The values dictionary with all single-valued boxes assigned

    Notes
    -----
    You should be able to complete this function by copying your code from the classroom
    """
    def confirm_digit(values, unit_keys, digit):
        for key in unit_keys:
            if digit in values[key]:
                values[key] = digit
        return values

    all_digits = '123456789'

    for unit_keys in unitlist:
        unit_digits = ''.join([values[box] for box in unit_keys])
        for digit in all_digits:
            if unit_digits.count(digit) == 1:
                values = confirm_digit(values, unit_keys, digit)
    return values


def reduce_puzzle(values):
    """Reduce a Sudoku puzzle by repeatedly applying all constraint strategies

    Parameters
    ----------
    values(dict)
        a dictionary of the form {'box_name': '123456789', ...}

    Returns
    -------
    dict or False
        The values dictionary after continued application of the constraint strategies
        no longer produces any changes, or False if the puzzle is unsolvable 
    """
    stalled = False
    while not stalled:
        solved_values_before = len([box for box in values.keys() if len(values[box]) == 1])
        values = eliminate(values)
        values = only_choice(values)
        solved_values_after = len([box for box in values.keys() if len(values[box]) == 1])
        stalled = solved_values_before == solved_values_after
        # Sanity check, return False if there is a box with zero available values:
        if len([box for box in values.keys() if len(values[box]) == 0]):
            return False
    return values


def search(values):
    """Apply depth first search to solve Sudoku puzzles in order to solve puzzles
    that cannot be solved by repeated reduction alone.

    Parameters
    ----------
    values(dict)
        a dictionary of the form {'box_name': '123456789', ...}

    Returns
    -------
    dict or False
        The values dictionary with all boxes assigned or False

    Notes
    -----
    You should be able to complete this function by copying your code from the classroom
    and extending it to call the naked twins strategy.
    """
    # First, reduce the puzzle using the previous function
    values = reduce_puzzle(values)
    if not values:
        return False
    values = naked_twins(values)
    if not values:
        return False
    # Choose one of the unfilled squares with the fewest possibilities
    num_choices = {(box, len(v)) for box, v in values.items() if len(v) > 1}
    if num_choices == set():
        return values
    min_box = min(num_choices, key=lambda x: x[1])[0]
    # Now use recursion to solve each one of the resulting sudokus, and if one returns a value (not False),
    # return that answer!
    for value in values[min_box]:
        temp_values = values.copy()
        temp_values[min_box] = value
        temp_values = search(temp_values)
        if temp_values:
            return temp_values


def solve(grid):
    """Find the solution to a Sudoku puzzle using search and constraint propagation

    Parameters
    ----------
    grid(string)
        a string representing a sudoku grid.
        
        Ex. '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'

    Returns
    -------
    dict or False
        The dictionary representation of the final sudoku grid or False if no solution exists.
    """
    values = grid2values(grid)
    values = search(values)
    return values


if __name__ == "__main__":
    diag_sudoku_grid = '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    display(grid2values(diag_sudoku_grid))
    result = solve(diag_sudoku_grid)
    display(result)

    try:
        import PySudoku
        PySudoku.play(grid2values(diag_sudoku_grid), result, history)

    except SystemExit:
        pass
    except:
        print('We could not visualize your board due to a pygame issue. Not a problem! It is not a requirement.')
