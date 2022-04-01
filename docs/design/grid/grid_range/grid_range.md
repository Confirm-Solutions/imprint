# Grid Range

In general, `kevlar` requires a notion of "a set of grid-points"
on which to simulate a given model.
As it turns out, we require a few more additional detail
to be able to integrate grid range into the framework.
This document covers the specification and the API
for our grid range class.

## Overview

A typical workflow of using a grid range is described below:
```mermaid
flowchart TB
    user([User])

    %% Grid range construction
    subgraph gr_subg [Grid Range Construction]
        get_gridpts[Create a list of grid-points]
        get_radii[Create a list of radii]
        get_ss[Create a list of simulation sizes]
        get_tiles[Create tiles]
        prune[Prune the grid range]
        
        get_gridpts --> get_tiles
        get_radii --> get_tiles
        get_ss --> get_tiles
        get_tiles --> prune
    end
    user --> gr_subg

    gr_subg --> process([Use grid range])
```

The following sections discuss in further detail
the subroutines depicted as rectangular tiles in the diagram above.

## Grid Range Specification

This section covers the required specification 
of a grid range concept.
Throughout this section, we will illustrate many concepts
using a running example of a user-defined grid-space.

### Grid-points

The first and foremost requirement is to store a list of grid-points.
As mentioned in [Grid Range](#grid-range),
this is the set of grid-points under which we simulate a given model.
An example is shown below with blue dots representing the grid-points
and the gray-space representing the grid-space of interest:

<img src="img_002.png" width="100" />

Note that the context, or meaning, of these points is defined by the model of interest.
See [Model](../model/model.md#attaching-gridrange) for more detail.
The user is responsible for constructing a valid list of grid-points
that adhere to the convention of the model of interest.

However, regardless of the context, 
a range of grid-points is still a meaningful quantity 
for the rest of the framework.
The framework only ever assumes that the grid-points lie in the space in which
we apply the Taylor expansion of the function of interest (e.g. Type I error function)
(see [UpperBound](../../math/stats/upper_bound/doc.pdf)).

### Radii

In [UpperBound](../../math/stats/upper_bound/doc.pdf),
the Type I error guarantees originate from having control of a Taylor expansion
around a small region `R` associated with each grid-point.
While it is true that `R` need not contain the corresponding grid-point,
accuracy is improved when it does. 
So, if a user has a grid-space they wish to get Type I error guarantees on,
we will assume that the space has been partitioned by a set of hypercubes
where each hypercube is defined by a grid-point as in [Grid-points](#grid-points)
as the center and a radii vector that defines the radius along each direction.

## Simulation Sizes

## Tiles

See [TODO: Tile](tile.md) for more detail.

TODO:
- tiles
- gridpoints
- meaning of gridpoints (delegate to model.md)
