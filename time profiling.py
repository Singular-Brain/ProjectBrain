from run import *


def main():
    import cProfile
    import pstats

    with cProfile.Profile() as pr:
        reward.set_label(0)
        model.run(progress_bar = True)

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats(filename='needs_profiling.prof')


if __name__ == '__main__':
    main()