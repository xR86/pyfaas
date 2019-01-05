### Config ###

CYCLE_HRS = 24

SUBDIVISION_MINS = 5

PEAKS = ['8:30',
         '19:30']

PEAK_DURATIONS_MINS = [120,
                       60]
PEAK_MAX_VARIATION_MINS = 120

MIN_TRAFFIC_PER_MINUTE = 1
MAX_TRAFFIC_PER_MINUTE = 10

MAX_TRAFFIC_PER_MINUTE_AT_PEAK = 100

##############

POLLING_INTERVAL = SUBDIVISION_MINS

CYCLE_MINS = CYCLE_HRS * 60

TIME_STEPS = list(zip(*[(mins // SUBDIVISION_MINS, f'{mins // 60:02d}:{mins % 60:02d}')
                        for mins
                        in range(0, CYCLE_MINS, CYCLE_MINS // 24)]))

TIME_STEPS = list(map(list, TIME_STEPS))

TIME_STEPS[0] += [TIME_STEPS[0][-1] + TIME_STEPS[0][1]]
TIME_STEPS[1] += ['24:00']
