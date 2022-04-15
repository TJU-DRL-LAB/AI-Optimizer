import argparse


def parse_arguments():
    parser = argparse.ArgumentParser("Easy MARL")
    parser.add_argument("--env-name", type=str, default="discrete_meeting", help='select an environment name')
    parser.add_argument("--scenario-name", type=str, default="", help='select an environment name')
    parser.add_argument("--agent-name", type=str, default="IDQN", help="select an agent name")
    temp_args = parser.parse_args()

    temp_args.exp_name = temp_args.env_name + "_" + temp_args.agent_name

    if temp_args.exp_name == "discrete_meeting_IDQN":
        from hyperparameters.discrete_meeting_IDQN import Hyperparameter
    elif temp_args.exp_name == "discrete_meeting_VDN":
        from hyperparameters.discrete_meeting_VDN import Hyperparameter
    elif temp_args.exp_name == "discrete_meeting_QMIX":
        from hyperparameters.discrete_meeting_QMIX import Hyperparameter
    elif temp_args.exp_name == "discrete_meeting_IPPO":
        from hyperparameters.discrete_meeting_IPPO import Hyperparameter
    elif temp_args.exp_name == "discrete_meeting_MAPPO":
        from hyperparameters.discrete_meeting_MAPPO import Hyperparameter

    # default scenario: Switch4-v0
    elif temp_args.exp_name == "discrete_magym_IDQN":
        from hyperparameters.discrete_magym_IDQN import Hyperparameter
    elif temp_args.exp_name == "discrete_magym_VDN":
        from hyperparameters.discrete_magym_VDN import Hyperparameter
    elif temp_args.exp_name == "discrete_magym_QMIX":
        from hyperparameters.discrete_magym_QMIX import Hyperparameter
    elif temp_args.exp_name == "discrete_magym_IPPO":
        from hyperparameters.discrete_magym_IPPO import Hyperparameter
    elif temp_args.exp_name == "discrete_magym_MAPPO":
        from hyperparameters.discrete_magym_MAPPO import Hyperparameter

    # you should set the parameters for continuous_meeting
    elif temp_args.exp_name == "continuous_meeting_IDDPG":
        from hyperparameters.continuous_meeting_IDDPG import Hyperparameter
    elif temp_args.exp_name == "continuous_meeting_MADDPG":
        from hyperparameters.continuous_meeting_MADDPG import Hyperparameter
    elif temp_args.exp_name == "continuous_meeting_IPPO":
        from hyperparameters.continuous_meeting_IPPO import Hyperparameter
    elif temp_args.exp_name == "continuous_meeting_MAPPO":
        from hyperparameters.continuous_meeting_MAPPO import Hyperparameter

    # default scenario: simple_tag
    elif temp_args.exp_name == "continuous_mpe_IDDPG":
        from hyperparameters.continuous_mpe_IDDPG import Hyperparameter
    elif temp_args.exp_name == "continuous_mpe_MADDPG":
        from hyperparameters.continuous_mpe_MADDPG import Hyperparameter
    elif temp_args.exp_name == "continuous_mpe_IPPO":
        from hyperparameters.continuous_mpe_IPPO import Hyperparameter
    elif temp_args.exp_name == "continuous_mpe_MAPPO":
        from hyperparameters.continuous_mpe_MAPPO import Hyperparameter
    else:
        raise NotImplementedError("{} does not exist.".format(temp_args.exp_name))

    if temp_args.env_name == "discrete_meeting" or temp_args.env_name == "continuous_meeting":
        # discrete/continuous_meeting don't need param "scenario_name"
        args = Hyperparameter()
        args.exp_name = temp_args.env_name + "_" + temp_args.agent_name
    else:
        args = Hyperparameter(temp_args.scenario_name)
        args.exp_name = temp_args.env_name + "_" + args.scenario_name + "_" + temp_args.agent_name

    return args
