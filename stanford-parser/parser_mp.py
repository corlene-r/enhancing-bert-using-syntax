import os
import time
import asyncio
from queue import Queue
from pathlib import Path
from argparse import ArgumentParser
from subprocess import Popen
from tqdm import tqdm

def ceildiv(a, b):
    """Performs ceiling division."""
    return -(a // -b)

def main():
    try:
        cpu_count = os.cpu_count()
    except NotImplementedError:
        cpu_count = 1

    # Define and then parse arguments
    arg_parser = ArgumentParser(prog="StanfordParser")
    arg_parser.add_argument('input_file', type=str, help="The file to output the results to.")
    arg_parser.add_argument('output_file', type=str, help="The file to output the results to.")
    arg_parser.add_argument('-s', '--subprocesses', type=int, default=max(cpu_count - 2, 1), help="The number of subprocesses to spawn to complete the task.")
    arg_parser.add_argument('-c', '--chunk_size', type=int, default=100, help="How many sentences to have each process turn into trees.")
    arg_parser.add_argument('-sl', '--sleep', type=int, default=1, help="How many seconds to sleep between process checks.")
    args = arg_parser.parse_args()

    if args.input_file == args.output_file:
        print("Input file cannot be the same as the output file!")
        return

    temps_folder = os.path.join('.', '__parser_temps__')

    # make temps folder if necessary
    Path(temps_folder).mkdir(parents=True, exist_ok=True)

    # figure out how many lines the file has
    num_lines = 0
    with open(args.input_file, 'r', encoding='utf8') as file:
        while True:
            line = file.readline()
            if len(line) == 0: break
            num_lines += 1

    # pre-define the task args
    tasks = []
    chunk_size = args.chunk_size
    num_lines = num_lines + (chunk_size - (num_lines % chunk_size))
    for i in range(num_lines // chunk_size):
        i_file: str = os.path.basename(args.output_file).replace('.', '_').replace('\\', '_').replace('/', '_')
        tasks.append((
            args.input_file,
            os.path.join(temps_folder, f'{i_file}_part_{i + 1}.tsv'),
            i * chunk_size,
            (i + 1) * chunk_size
        ))

    all_tasks = list(tasks)
    tasks = list(reversed(tasks))

    start_time = time.time()
    curr_start = start_time

    # run the processes to completion
    running_tasks: list[Popen] = []
    with tqdm(total=len(tasks), desc=f"Turning into trees {len(tasks)} chunk(s) of {chunk_size} sentences each") as pbar:
        while (len(tasks) > 0) or (len(running_tasks) > 0):

            popped_off = 0

            # pop off all finished tasks
            for i in reversed(range(len(running_tasks))):
                if running_tasks[i].poll() is None: continue
                popped_off += 1
                running_tasks.pop(i)

            # add any new tasks (when possible)
            while len(tasks) > 0 and len(running_tasks) < args.subprocesses:
                (input_file, output_file, start_line, end_line) = tasks.pop()
                pargs = ['python', os.path.join('.', 'parser.py'), input_file, output_file, f'-s {start_line}', f'-e {end_line}']
                running_tasks.append(Popen(pargs))

            # update the progress bar
            if popped_off > 0:
                pbar.update(popped_off)

            # sleep to give tasks time to run
            if len(running_tasks) > 0:
                time.sleep(args.sleep)

    print(f'\nProcessing {args.input_file} took {(((time.time() - curr_start) // 60) // 60)} hours, {((time.time() - curr_start) // 60)} minutes, and {(time.time() - curr_start) % 60} seconds')

    # put the contents of the parts file into one larger file (the output file)
    with open(args.output_file, 'w+', encoding='utf8') as out:
        for task in tqdm(all_tasks, desc="Appending each task's output to " + args.output_file):
            (_, output_file, _, _) = task

            with open(output_file, 'r', encoding='utf8') as inp:
                out.write(inp.read())

    print(f'\nWriting to {args.output_file} took {(((time.time() - curr_start) // 60) // 60)} hours, {((time.time() - curr_start) // 60)} minutes, and {(time.time() - curr_start) % 60} seconds')

if __name__ == "__main__":
    main()
