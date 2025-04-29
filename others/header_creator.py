# -*- coding: utf-8 -*-
# -*- authors : Filippo Quadri -*-
# -*- date : 2024-06-17-*-
# -*- Last revision: 2025-04-25 by roduit -*-
# -*- python version : 3.9.19 -*-
# -*- Description: Function to create header *-


def create_header(name, length=86):
    """Create a header for the file

    Args:
        * name (str): The name of the section

        * length (int): The length of the header
    """
    print(f"# {'='*length}")
    padding = length - len(name) - 2 * 5
    padding_left = padding // 2
    padding_right = padding - padding_left
    print(f"# {'='*5}{' '*padding_left}{name}{' '*padding_right}{'='*5}")
    print(f"# {'='*length}")


create_header("MODEL PARAMS")
