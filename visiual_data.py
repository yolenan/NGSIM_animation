import os
import numpy as np
import pandas as pd
import pyglet

path = os.getcwd()


class Routing(object):
    viewer = None

    def __init__(self):
        df = pd.read_csv(path + '/data/trajectories-0400-0415.txt', header=None, sep='\s+', skiprows=[0])
        df[4] = df[4] + 50
        # print(min(df[4]), max(df[4]), min(df[5]), max(df[5]), min(df[6]), max(df[6]), min(df[7]), max(df[7]),
        #       min(df[8]), max(df[8]), min(df[9]), max(df[9]))
        self.car_data = df[0].drop_duplicates().reset_index(drop=True).tolist()
        self.all_car = []
        for item in self.car_data:
            self.all_car.append(df[df[0] == item].reset_index(drop=True))
        self.endtime = max(df[1])
        self.t = [0 for k in range(len(self.car_data))]
        self.car_pos = np.zeros(len(self.car_data),
                                dtype=[('vehicle_id', np.int), ('frame_id', np.int), ('total_frame', np.int),
                                       ('x', np.float16), ('y', np.float16), ('vehicle_L', np.float16),
                                       ('vehicle_W', np.float16), ('type', np.int), ('num', np.int)])
        self.car_pos['vehicle_id'] = [item for item in self.car_data]
        self.car_pos['frame_id'] = [self.all_car[i].at[0, 1] for i in range(len(self.car_data))]
        self.car_pos['total_frame'] = [self.all_car[i].at[0, 2] for i in range(len(self.car_data))]
        self.car_pos['x'] = [self.all_car[i].at[0, 4] for i in range(len(self.car_data))]
        self.car_pos['y'] = [self.all_car[i].at[0, 5] for i in range(len(self.car_data))]
        self.car_pos['vehicle_L'] = [0] * len(self.car_data)
        self.car_pos['vehicle_W'] = [0] * len(self.car_data)
        self.car_pos['type'] = [self.all_car[i].at[0, 10] for i in range(len(self.car_data))]
        self.car_pos['num'] = [0] * len(self.car_data)
        self.global_time = 0

    def reset(self):
        pass

    def step(self, action):
        global timegap
        done = False
        s = self.car_pos
        self.global_time = self.global_time + action
        if self.global_time >= self.endtime:
            done = True
        for j in range(len(self.car_data)):
            # pos update
            if self.global_time >= self.car_pos['frame_id'][j]:
                if self.car_pos['num'][j] < self.car_pos['total_frame'][j] - 1:
                    self.car_pos['frame_id'][j] = self.all_car[j].at[self.car_pos['num'][j] + 1, 1]
                    self.car_pos['x'][j] = self.all_car[j].at[self.car_pos['num'][j] + 1, 4]
                    self.car_pos['y'][j] = self.all_car[j].at[self.car_pos['num'][j] + 1, 5]
                    self.car_pos['vehicle_L'][j] = self.all_car[j].at[self.car_pos['num'][j] + 1, 8]
                    self.car_pos['vehicle_W'][j] = self.all_car[j].at[self.car_pos['num'][j] + 1, 9]
                    if self.car_pos['y'][j] + self.car_pos['vehicle_L'][j] / 2 > 495:
                        self.car_pos['vehicle_L'][j] = max((495 - self.car_pos['y'][j]) * 2, 0)
                    elif self.car_pos['y'][j] - self.car_pos['vehicle_L'][j] / 2 < 50:
                        self.car_pos['vehicle_L'][j] = max((self.car_pos['y'][j] - 50) * 2, 0)
                    self.car_pos['num'][j] += 1
        r = 0
        return s, r, done

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(self.car_pos, self.global_time)
        self.viewer.render(self.global_time)

    def sample_action(self):
        return 1


class Viewer(pyglet.window.Window):
    node_size = 10
    car_size = 2

    def __init__(self, car_pos, global_time):
        super(Viewer, self).__init__(width=300, height=600, resizable=False, caption='Avmap', vsync=False)
        pyglet.gl.glClearColor(1, 1, 1, 1)
        self.car_pos = car_pos
        self.batch = pyglet.graphics.Batch()
        self.time = global_time
        line_from_x = [50, 50, 132, 144, 50, 62, 74, 86, 98, 110, 132]
        line_to_x = [132, 144, 144, 144, 50, 62, 74, 86, 98, 110, 132]
        line_from_y = [50, 500, 320, 320, 50, 50, 50, 50, 50, 50, 50]
        line_to_y = [50, 500, 320, 500, 500, 500, 500, 500, 500, 500, 500]
        line_list = []
        for j in range(len(line_from_x)):
            from_x = line_from_x[j]
            from_y = line_from_y[j]
            to_x = line_to_x[j]
            to_y = line_to_y[j]
            line_list.append(from_x)
            line_list.append(from_y)
            line_list.append(to_x)
            line_list.append(to_y)
        dim = int(len(line_list) / 2)
        self.line = self.batch.add(
            dim, pyglet.gl.GL_LINES, None,
            ('v2f', line_list),
            ('c3B', (50, 129, 249) * dim)
        )
        # draw cars
        self.car = []
        self.init_x = car_pos['x'].copy()
        self.init_y = car_pos['y'].copy()
        for t in range(len(self.car_pos['x'])):
            carx = self.car_pos['x'][t]
            cary = self.car_pos['y'][t]
            carx_size = self.car_pos['vehicle_W'][t] / 2
            cary_size = self.car_pos['vehicle_L'][t] / 2
            if self.car_pos['type'][t] == 2:
                col = (105, 105, 105)
            else:
                col = (252, 157, 154)
            self.car.append(self.batch.add(
                4, pyglet.gl.GL_QUADS, None,  # 4 corners
                ('v2f', [carx - carx_size, cary + cary_size,  # location
                         carx + carx_size, cary + cary_size,
                         carx + carx_size, cary - cary_size,
                         carx - carx_size, cary - cary_size]),
                ('c3B', col * 4)))  # color

        # draw time window
        t = str(self.time)
        self.name = pyglet.text.Label(text='NGSIM Animation by Y.A.',
                                      font_name='Times New Roman',
                                      x=10,
                                      y=550,
                                      batch=self.batch,
                                      font_size=18,
                                      color=(0, 0, 0, 200))
        self.time_label = pyglet.text.Label(text='Frame ID',
                                            font_name='Times New Roman',
                                            x=170,
                                            y=200,
                                            batch=self.batch,
                                            font_size=16,
                                            color=(0, 0, 0, 200))
        self.car_label = pyglet.text.Label(text='Car Num',
                                           font_name='Times New Roman',
                                           x=170,
                                           y=100,
                                           batch=self.batch,
                                           font_size=16,
                                           color=(0, 0, 0, 200))
        self.system_time = pyglet.text.Label(text=t,
                                             font_name='Times New Roman',
                                             x=170,
                                             y=150,
                                             batch=self.batch,
                                             font_size=16,
                                             color=(0, 0, 0, 200))
        self.car_num = pyglet.text.Label(text=str(0),
                                         font_name='Times New Roman',
                                         x=170,
                                         y=50,
                                         batch=self.batch,
                                         font_size=16,
                                         color=(0, 0, 0, 200))

    def render(self, gtime):
        self.time = gtime
        self._update_veh()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        self.clear()
        self.batch.draw()

    def _update_veh(self):
        self.system_time.text = str(round(self.time, 3))
        car_num = 0
        for t in range(len(self.car_pos['x'])):
            car_x = self.car_pos['x'][t]
            car_y = self.car_pos['y'][t]
            carx_size = self.car_pos['vehicle_W'][t] / 2
            cary_size = self.car_pos['vehicle_L'][t] / 2
            # print(self.init_x[t], car_x)
            if car_x != self.init_x[t] or car_y != self.init_y[t]:
                xy01 = np.array([car_x - carx_size, car_y + cary_size])
                xy02 = np.array([car_x + carx_size, car_y + cary_size])
                xy03 = np.array([car_x + carx_size, car_y - cary_size])
                xy04 = np.array([car_x - carx_size, car_y - cary_size])
                if car_x >= 0 and car_x <= 200 and car_y >= 0 and car_y <= 500:
                    car_num += 1
                    self.car[t].vertices = np.concatenate((xy01, xy02, xy03, xy04))
        self.car_num.text = str(car_num)


if __name__ == '__main__':
    env = Routing()
    done = False
    while not done:
        env.render()
        s, r, done = env.step(env.sample_action())
