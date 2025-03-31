# 0 - 4 parameters (`'gausshyper'`)
distcont: list[
    tuple[
        str,
        tuple[()]
        | tuple[float | int | bool]
        | tuple[float | int | bool, float | int | bool]
        | tuple[float | int | bool, float | int | bool, float | int | bool]
        | tuple[float | int | bool, float | int | bool, float | int | bool, float | int | bool],
    ]
]

# 0 - 4 parameters (`'gausshyper'`)
invdistcont: list[
    tuple[
        str,
        tuple[()]
        | tuple[float | int | bool]
        | tuple[float | int | bool, float | int | bool]
        | tuple[float | int | bool, float | int | bool, float | int | bool]
        | tuple[float | int | bool, float | int | bool, float | int | bool, float | int | bool],
    ]
]

# 1 - 4 parameters (`'nchypergeom_fisher'` and `'nchypergeom_wallenius'`)
distdiscrete: list[
    tuple[
        str,
        tuple[float | int | bool]
        | tuple[float | int | bool, float | int | bool]
        | tuple[int | bool, float | int | bool, float | int | bool]
        | tuple[int | bool, int | bool, int | bool, float | int | bool],
    ]
]
# 1 - 4 parameters (`'nchypergeom_fisher'` and `'nchypergeom_wallenius'`)
invdistdiscrete: list[
    tuple[
        str,
        tuple[float | int | bool]
        | tuple[float | int | bool, float | int | bool]
        | tuple[int | bool, float | int | bool, float | int | bool]
        | tuple[int | bool, int | bool, int | bool, float | int | bool],
    ]
]
