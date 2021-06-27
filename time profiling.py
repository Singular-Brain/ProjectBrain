from run import main as run_main
import cProfile
import pstats

def main() -> None:
    with cProfile.Profile() as pr:
        run_main()

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats(filename='needs_profiling.prof')


if __name__ == '__main__':
    main()