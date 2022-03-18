import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # add agents
        world.agents = [Agent() for i in range(1)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True
        # add landmarks
        num_landmarks = 3
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25, 0.25, 0.25])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.75, 0.75, 0.75])
        world.landmarks[0].color = np.array([0.75, 0.25, 0.25])
        world.landmarks[1].color = np.array([0.75, 0.75, 0.25])
        world.landmarks[2].color = np.array([0.6, 0.6, 0.6])

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.catch_flag = 0
            agent.target_distance = 0.3

        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        dist2 = np.sum(np.square(agent.state.p_pos - world.landmarks[0].state.p_pos))
        dist3 = np.sum(np.square(agent.state.p_pos - world.landmarks[1].state.p_pos))
        dist4 = np.sum(np.square(agent.state.p_pos - world.landmarks[2].state.p_pos))


        if (agent.action.u[3] == 1):
            if (np.sum(
                    np.square(agent.state.p_pos - world.landmarks[
                        0].state.p_pos)) <= agent.target_distance ** 2) and agent.catch_flag == 0:
                dist2 = dist2 - 10
                agent.catch_flag = 1
                # print('位置正确')
            elif (np.sum(
                    np.square(agent.state.p_pos - world.landmarks[
                        1].state.p_pos)) <= agent.target_distance ** 2) and agent.catch_flag == 1:
                dist3 = dist3 - 10
                agent.catch_flag = 2
            elif (np.sum(
                    np.square(agent.state.p_pos - world.landmarks[
                        2].state.p_pos)) <= agent.target_distance ** 2) and agent.catch_flag == 2:
                dist4 = dist4 - 20
            # else:
            #     if agent.catch_flag == 0:
            #         dist2 = dist2 + 0.05
            #     elif agent.catch_flag == 1 :
            #         dist3 = dist3 + 0.05
            #     else:
            #         dist4 = dist4 + 0.05

        if  agent.catch_flag == 0:
            return -dist2
        elif  agent.catch_flag == 1:
            return -dist3
        else:
            return -dist4

        # agent.catch_flag = 0



    # def reward(self, agent, world):
    #     dist2 = np.sum(np.square(agent.state.p_pos - world.landmarks[0].state.p_pos))
    #     if(agent.action.u[3]==1):
    #         if(np.sum(np.square(agent.state.p_pos - world.landmarks[0].state.p_pos))<=agent.target_distance**2):
    #             dist2=dist2-5
    #             # print('catch 成功')
    #         else:
    #             dist2 = dist2 + 0.5
    #     return -dist2

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        # for entity in world.landmarks:
        #     entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        if agent.catch_flag == 0:
            entity_pos.append(world.landmarks[0].state.p_pos - agent.state.p_pos)
        elif agent.catch_flag == 1:
            entity_pos.append(world.landmarks[1].state.p_pos - agent.state.p_pos)
        else:
            entity_pos.append(world.landmarks[2].state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + entity_pos)
