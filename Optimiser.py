import casadi


def select_squad(data):
    num_players = len(data)
    players = list(data.keys())
    teams = ['team1', 'team2', 'team3', 'team4']
    x = casadi.SX.sym('x', num_players)
    y = casadi.SX.sym('y')

    f = 0

    for i in range(num_players):
        f += x[i] * data[players[i]]['prediction']

    f = -f

    g = casadi.SX()

    lbg = []
    ubg = []

    #max three players from each team
    for i in range(len(teams)):
        g_int = 0
        for j in range(num_players):
            if data[players[j]]['team'] == teams[i]:
                g_int += x[j]
        g_int -= 3

        g = casadi.vertcat(g, g_int)
        lbg.append(-casadi.inf)
        ubg.append(0)

    #max two free changes (y is greater than or equal to 0 and y is greater than or equal to number of changes - 2)

    g_y = y

    for i in range(num_players):
        if data[players[i]]['currently_selected'] is True:
            g_y += x[i]

    g_y -= 11
    g_y += 2

    g = casadi.vertcat(g, g_y)

    lbg.append(0)
    ubg.append(casadi.inf)

    #1gk, 3 def, 4 mid, 3 att

    g_gk = 0

    for i in range(num_players):
        if data[players[i]]['position'] == 'gk':
            g_gk += x[i]
    g_gk += -1

    g_def = 0

    for i in range(num_players):
        if data[players[i]]['position'] == 'def':
            g_def += x[i]
    g_def += -3

    g_mid = 0

    for i in range(num_players):
        if data[players[i]]['position'] == 'mid':
            g_mid += x[i]
    g_mid += -4

    g_att = 0

    for i in range(num_players):
        if data[players[i]]['position'] == 'att':
            g_att += x[i]
    g_att += -3

    g = casadi.vertcat(g, g_gk, g_def, g_mid, g_att)

    lbg.append(0)
    ubg.append(0)
    lbg.append(0)
    ubg.append(0)
    lbg.append(0)
    ubg.append(0)
    lbg.append(0)
    ubg.append(0)

    # Cost of players should be less than the budget (100)

    g_cost = 0
    for i in range(num_players):
        g_cost += x[i] * data[players[i]]['cost']

    g_cost += 2 * y
    g_cost -= 100

    g = casadi.vertcat(g, g_cost)

    lbg.append(-casadi.inf)
    ubg.append(0)

    discrete_list = [True] * (num_players + 1)
    lower_bounds = [0] * (num_players + 1)
    upper_bounds = [1] * num_players
    upper_bounds.append(casadi.inf)

    problem = {'x': casadi.vertcat(x, y), 'f': f, 'g': g}
    solver = casadi.nlpsol('nlp_solver', 'bonmin', problem, {'discrete': discrete_list})
    solution = solver(lbx=lower_bounds, ubx=upper_bounds, lbg=lbg, ubg=ubg)

    squad = []

    for i in range(num_players):
        if float(solution['x'][i]) == 1:
            squad.append(players[i])

    return squad


# Just some test data
info = {
    'player1': {'prediction': 2, 'position': 'gk', 'team': 'team1', 'currently_selected': True, 'cost': 8},
    'player2': {'prediction': 4, 'position': 'gk', 'team': 'team3', 'currently_selected': False, 'cost': 11},
    'player3': {'prediction': 3, 'position': 'def', 'team': 'team2', 'currently_selected': True, 'cost': 9},
    'player4': {'prediction': 4, 'position': 'def', 'team': 'team3', 'currently_selected': True, 'cost': 9},
    'player5': {'prediction': 1, 'position': 'def', 'team': 'team4', 'currently_selected': True, 'cost': 6},
    'player6': {'prediction': 7, 'position': 'def', 'team': 'team1', 'currently_selected': False, 'cost': 15},
    'player7': {'prediction': 7, 'position': 'mid', 'team': 'team1', 'currently_selected': True, 'cost': 9},
    'player8': {'prediction': 2, 'position': 'mid', 'team': 'team2', 'currently_selected': True, 'cost': 8},
    'player9': {'prediction': 1, 'position': 'mid', 'team': 'team3', 'currently_selected': True, 'cost': 5},
    'player10': {'prediction': 6, 'position': 'mid', 'team': 'team4', 'currently_selected': True, 'cost': 10},
    'player11': {'prediction': 4, 'position': 'mid', 'team': 'team1', 'currently_selected': False, 'cost': 12},
    'player12': {'prediction': 2, 'position': 'att', 'team': 'team2', 'currently_selected': True, 'cost': 10},
    'player13': {'prediction': 2, 'position': 'att', 'team': 'team3', 'currently_selected': True, 'cost': 9},
    'player14': {'prediction': 4, 'position': 'att', 'team': 'team4', 'currently_selected': False, 'cost': 7},
    'player15': {'prediction': 6, 'position': 'att', 'team': 'team1', 'currently_selected': True, 'cost': 9},
    'player16': {'prediction': 3, 'position': 'gk', 'team': 'team2', 'currently_selected': False, 'cost': 11},
    'player17': {'prediction': 1, 'position': 'gk', 'team': 'team4', 'currently_selected': False, 'cost': 8},
    'player18': {'prediction': 5, 'position': 'def', 'team': 'team2', 'currently_selected': False, 'cost': 16},
    'player19': {'prediction': 3, 'position': 'def', 'team': 'team3', 'currently_selected': False, 'cost': 12},
    'player20': {'prediction': 2, 'position': 'mid', 'team': 'team2', 'currently_selected': False, 'cost': 9},
    'player21': {'prediction': 2, 'position': 'mid', 'team': 'team3', 'currently_selected': False, 'cost': 8},
    'player22': {'prediction': 6, 'position': 'att', 'team': 'team2', 'currently_selected': False, 'cost': 12},
    'player23': {'prediction': 4, 'position': 'att', 'team': 'team3', 'currently_selected': False, 'cost': 11},
    'player24': {'prediction': 5, 'position': 'mid', 'team': 'team4', 'currently_selected': False, 'cost': 10},
    'player25': {'prediction': 1, 'position': 'gk', 'team': 'team3', 'currently_selected': False, 'cost': 3},
    'player26': {'prediction': 1, 'position': 'def', 'team': 'team2', 'currently_selected': False, 'cost': 3},
    'player27': {'prediction': 2, 'position': 'mid', 'team': 'team1', 'currently_selected': False, 'cost': 3},
    'player28': {'prediction': 1, 'position': 'att', 'team': 'team4', 'currently_selected': False, 'cost': 3}
}

new_team = select_squad(info)

print(new_team)
