from pykevlar.core import GridRange


class SimpleBatchIter():
    def __init__(self,
                 simple_batch):
        self.simple_batch = simple_batch
        self.pos = 0

    def __next__(self):
        sb = self.simple_batch

        if self.pos == sb.grid_range.size():
            raise StopIteration

        size = min(sb.max_size,
                   sb.grid_range.size()-self.pos)

        gr = GridRange(sb.grid_range.dim(),
                       size)

        # copy over thetas
        thetas = gr.get_thetas()
        big_thetas = sb.grid_range.get_thetas()
        thetas[...] = big_thetas[:, self.pos:(self.pos+size)]

        # copy over radii
        radii = gr.get_radii()
        big_radii = sb.grid_range.get_radii()
        radii[...] = big_radii[:, self.pos:(self.pos+size)]

        # Assumptions:
        # - the sim_sizes are fixed for all gridpoints
        #   so just grab one of the elements.
        # - each batch will process the full sim_size.
        #   so no need to do anything special for sim_size_rem.
        sim_size = sb.grid_range.get_sim_sizes_const()[0]

        self.pos += size

        return gr, sim_size


class SimpleBatch():

    def __init__(self, grid_range, max_size):
        if max_size == 0:
            raise ValueError("max_size must be either positive or negative.")

        self.grid_range = grid_range
        self.max_size = max_size if max_size > 0 else grid_range.size()

    def __iter__(self):
        return SimpleBatchIter(self)
