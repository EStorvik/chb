
from dolfinx import plot

import pyvista as pv
import pyvistaqt as pvqt


if pv.OFF_SCREEN:
    pv.start_xvfb(wait=0.5)

class PyvistaVizualization:


    def __init__(self, V, xi, t0, name = "phase-field") -> None:
        """
        Initialize the visualization object

        Args:
            V (FunctionSpace): Function space. Provide the function space of the solution that you want to 
                                plot. F.ex. use .subs(0) to get the function space of the first component 
                                of the solution.
            xi (Function): The solution to visualize
            t0 (float): The initial time
            name (str): The name of the scalar field to plot
        """
        self.V0, self.dofs = V.collapse()
        self.name = name

        # Create a VTK 'mesh' with 'nodes' at the function dofs
        self.topology, self.cell_types, self.x = plot.vtk_mesh(self.V0)
        self.grid = pv.UnstructuredGrid(self.topology, self.cell_types, self.x)

        # Set output data
        self.grid.point_data[name] = xi.x.array[self.dofs].real
        self.grid.set_active_scalars(name)
        self.p = pvqt.BackgroundPlotter(title=self.name, auto_update=True)
        self.p.add_mesh(self.grid, clim=[0, 1])
        self.p.view_xy(True)
        self.p.add_text(f"time: {t0}", font_size=12, name="timelabel")


    def update(self, xi, t):
        """
        Update the visualization with the new solution xi at time t

        Args:
            xi (Function): The new solution
            t (float): The new time
        """
        self.p.add_text(f"time: {t:.2e}", font_size=12, name="timelabel")
        self.grid.point_data[self.name] = xi.x.array[self.dofs].real
        self.p.app.processEvents()
    


# Update ghost entries and plot
    def final_plot(self, xi):
        """
        Update the visualization with the new solution xi at time t

        Args:
            xi (Function): The new solution
            t (float): The new time
        """

        xi.x.scatter_forward()
        self.grid.point_data[self.name] = xi.x.array[self.dofs].real

        screenshot = None
        if pv.OFF_SCREEN:
            screenshot = {self.name}+".png"
        pv.plot(self.grid, show_edges=True, screenshot=screenshot)