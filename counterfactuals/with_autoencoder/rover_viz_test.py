import rover_domain
import numpy as np
import torch
import rendering
import time

rd = rover_domain.RoverDomain(2, 1, 1000)
viewer = rendering.Viewer(700, 700)

def render():

    viewer.geoms = []
    # add all the entities you want to render (POIs, agents, anything debug)
    for rover_id in range(rd.n_rovers):
        geom = rendering.make_circle(0.1)
        xform = rendering.Transform()
        color = [0.,1.,0.] if rover_id < 1 else [0.,0.,1.] 
        """
        @Connor, you can add this as an attribute in your agent class.
        self.color = np.array(
            [np.random.uniform(low=0.0, high=1),
             np.random.uniform(low=0.0, high=1),
             np.random.uniform(low=0.0, high=1)])
        """
        geom.set_color(*color)
        x, y = rd.rover_positions[rover_id, 0], rd.rover_positions[rover_id, 1]
        # print ("x,y: ",x,", ",y)
        xform.set_translation(x,y)
        geom.add_attr(xform)
        viewer.add_geom(geom)        

    for poi_id in range(rd.n_pois):
        geom = rendering.make_circle(0.1)
        xform = rendering.Transform()
        color = [0.,0.,0.]
        geom.set_color(*color)
        x, y = rd.poi_positions[poi_id, 0], rd.poi_positions[poi_id, 1]
        xform.set_translation(x,y)
        geom.add_attr(xform)
        viewer.add_geom(geom)
        
        # @Connor, for example, a debugging geom could be POI observation radius:
        observation_radius = 2.0
        geom = rendering.make_circle(observation_radius)
        geom.set_color(*color, alpha=0.05)
        xform = rendering.Transform()
        xform.set_translation(x,y)
        geom.add_attr(xform)
        viewer.add_onetime(geom)

    """
    @Connor you might want to create and store these geometries on the stack and just update the transform every frame, as opposed to creating them every frame like above
    """


    # Assuming world dimens
    w = 25
    h = 25
    # A debug geom for world bound
    world_bound_geom = rendering.make_polygon(
        [(0, 0), (w, 0), (w, h), (0, h)], False)
    world_bound_geom.set_color(0., 0., 0., alpha=0.2)
    xform = rendering.Transform()
    xform.set_translation(-w/2, -h/2)
    world_bound_geom.add_attr(xform)
    viewer.add_onetime(world_bound_geom)

    bound = 15 # viewport size
    viewer.set_bounds(-bound,bound,-bound,bound)

    # Use cam_range to zoom in/out
    # viewer.cam_range = 10

    viewer.render()


def main():
    done = False
    state = rd.rover_observations
    reward = 0
    while not done:
        state = np.array(state)
        state = torch.tensor(state.flatten())
        action = np.array([0, 1])
        render()
        time.sleep(0.05)
        state, r, done, _ = rd.step(np.array([[0, 1], [1, 1]], dtype='double'))
        reward += np.array(r)[0]
    return reward

main()
