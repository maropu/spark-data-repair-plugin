#!/usr/bin/env python3

#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import logging
from argparse import ArgumentParser
import os
import re
from shutil import rmtree, which
import subprocess
import sys
import tempfile
from threading import Thread, Lock
import time
import uuid
import queue as Queue
from multiprocessing import Manager


python_test_goals = [
    "repair.tests.test_costs",
    "repair.tests.test_detectors",
    "repair.tests.test_misc",
    "repair.tests.test_model",
    "repair.tests.test_model_perf",
    "repair.tests.test_utils"
]


def print_red(text):
    print('\033[31m' + text + '\033[0m')


SKIPPED_TESTS = None
LOG_FILE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "unit-tests.log")
FAILURE_REPORTING_LOCK = Lock()
LOGGER = logging.getLogger()


def run_individual_python_test(target_dir, test_name, pyspark_python, params):
    env = dict(os.environ)
    env.update({
        'PYTHONPATH': params["REPAIR_MODULE_PATH"],
        'REPAIR_API_LIB': params["REPAIR_API_LIB"],
        'REPAIR_TESTDATA': params["REPAIR_TESTDATA"],
        'SPARK_TESTING': '1',
        'SPARK_PREPEND_CLASSES': '1',
        'PYSPARK_PYTHON': which(pyspark_python),
        'PYSPARK_DRIVER_PYTHON': which(pyspark_python),
        # Preserve legacy nested timezone behavior for pyarrow>=2, remove after SPARK-32285
        'PYARROW_IGNORE_TIMEZONE': '1',
    })

    # Create a unique temp directory under 'target/' for each run. The TMPDIR variable is
    # recognized by the tempfile module to override the default system temp directory.
    tmp_dir = os.path.join(target_dir, str(uuid.uuid4()))
    while os.path.isdir(tmp_dir):
        tmp_dir = os.path.join(target_dir, str(uuid.uuid4()))
    os.mkdir(tmp_dir)
    env["TMPDIR"] = tmp_dir
    metastore_dir = os.path.join(tmp_dir, str(uuid.uuid4()))
    while os.path.isdir(metastore_dir):
        metastore_dir = os.path.join(metastore_dir, str(uuid.uuid4()))
    os.mkdir(metastore_dir)

    # Also override the JVM's temp directory by setting driver and executor options.
    java_options = "-Djava.io.tmpdir={0} -Dio.netty.tryReflectionSetAccessible=true".format(tmp_dir)
    spark_args = [
        "--conf", "spark.driver.extraJavaOptions='{0}'".format(java_options),
        "--conf", "spark.executor.extraJavaOptions='{0}'".format(java_options),
        "--conf", "spark.sql.warehouse.dir='{0}'".format(metastore_dir),
        "pyspark-shell"
    ]
    env["PYSPARK_SUBMIT_ARGS"] = " ".join(spark_args)

    LOGGER.info("Starting test(%s): %s", pyspark_python, test_name)
    start_time = time.time()
    try:
        per_test_output = tempfile.TemporaryFile()
        retcode = subprocess.Popen(
            ["pyspark"] + test_name.split(),
            stderr=per_test_output, stdout=per_test_output, env=env).wait()
        rmtree(tmp_dir, ignore_errors=True)
    except:
        LOGGER.exception("Got exception while running %s with %s", test_name, pyspark_python)
        # Here, we use os._exit() instead of sys.exit() in order to force Python to exit even if
        # this code is invoked from a thread other than the main thread.
        os._exit(1)
    duration = time.time() - start_time
    # Exit on the first failure.
    if retcode != 0:
        try:
            with FAILURE_REPORTING_LOCK:
                with open(LOG_FILE, 'ab') as log_file:
                    per_test_output.seek(0)
                    log_file.writelines(per_test_output)
                per_test_output.seek(0)
                for line in per_test_output:
                    decoded_line = line.decode("utf-8", "replace")
                    if not re.match('[0-9]+', decoded_line):
                        print(decoded_line, end='')
                per_test_output.close()
        except:
            LOGGER.exception("Got an exception while trying to print failed test output")
        finally:
            print_red("\nHad test failures in %s with %s; see logs." % (test_name, pyspark_python))
            # Here, we use os._exit() instead of sys.exit() in order to force Python to exit even if
            # this code is invoked from a thread other than the main thread.
            os._exit(-1)
    else:
        skipped_counts = 0
        try:
            per_test_output.seek(0)
            # Here expects skipped test output from unittest when verbosity level is
            # 2 (or --verbose option is enabled).
            decoded_lines = map(lambda line: line.decode("utf-8", "replace"), iter(per_test_output))
            skipped_tests = list(filter(
                lambda line: re.search(r'test_.* \(.*\) ... (skip|SKIP)', line),
                decoded_lines))
            skipped_counts = len(skipped_tests)
            if skipped_counts > 0:
                key = (pyspark_python, test_name)
                assert SKIPPED_TESTS is not None
                SKIPPED_TESTS[key] = skipped_tests
            per_test_output.close()
        except:
            import traceback
            print_red("\nGot an exception while trying to store "
                      "skipped test output:\n%s" % traceback.format_exc())
            # Here, we use os._exit() instead of sys.exit() in order to force Python to exit even if
            # this code is invoked from a thread other than the main thread.
            os._exit(-1)
        if skipped_counts != 0:
            LOGGER.info(
                "Finished test(%s): %s (%is) ... %s tests were skipped", pyspark_python, test_name,
                duration, skipped_counts)
        else:
            LOGGER.info(
                "Finished test(%s): %s (%is)", pyspark_python, test_name, duration)


def get_default_python_executables():
    python_execs = [x for x in ["python3.7", "pypy3"] if which(x)]

    if "python3.7" not in python_execs:
        p = which("python3")
        if not p:
            LOGGER.error("No python3 executable found.  Exiting!")
            os._exit(1)
        else:
            python_execs.insert(0, p)
    return python_execs


def parse_opts():
    parser = ArgumentParser(prog="run-tests")
    parser.add_argument(
        "--package", type=str, required=True,
        help="A compiled jar assembly of `spark-data-repair-plugin`"
    )
    parser.add_argument(
        "--module", type=str, required=True,
        help="A absolute path of the test repair module"
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="An absolute path of test data shared among all the test suties"
    )
    parser.add_argument(
        "--python-executables", type=str, default=','.join(get_default_python_executables()),
        help="A comma-separated list of Python executables to test against (default: %(default)s)"
    )
    parser.add_argument(
        "-p", "--parallelism", type=int, default=4,
        help="The number of suites to test in parallel (default %(default)d)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable additional debug logging"
    )

    group = parser.add_argument_group("Developer Options")
    group.add_argument(
        "--testnames", type=str,
        default=None,
        help=(
            "A comma-separated list of specific modules, classes and functions of doctest "
            "or unittest to test. "
            "For example, 'pyspark.sql.tests FooTests' to run the specific class of unittests, "
            "'pyspark.sql.tests FooTests.test_foo' to run the specific unittest in the class. ")
    )

    args, unknown = parser.parse_known_args()
    if unknown:
        parser.error("Unsupported arguments: %s" % ' '.join(unknown))
    if args.parallelism < 1:
        parser.error("Parallelism cannot be less than 1")
    return args


def main():
    opts = parse_opts()

    # Makes params for `spark-data-reiapr-plugin` tests
    params = {
        "REPAIR_API_LIB": opts.package,
        "REPAIR_MODULE_PATH": opts.module,
        "REPAIR_TESTDATA": opts.data
    }

    if opts.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    logging.basicConfig(stream=sys.stdout, level=log_level, format="%(message)s")
    LOGGER.info("Running PySpark tests. Output is in %s", LOG_FILE)
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    python_execs = opts.python_executables.split(',')
    LOGGER.info("Will test against the following Python executables: %s", python_execs)

    if opts.testnames is not None:
        testnames_to_test = opts.testnames.split(',')
        LOGGER.info("Will test the following Python tests: %s", testnames_to_test)

    task_queue = Queue.PriorityQueue()
    for python_exec in python_execs:
        python_implementation = subprocess.check_output(
            [python_exec, "-c", "import platform; print(platform.python_implementation())"],
            universal_newlines=True).strip()
        LOGGER.info("%s python_implementation is %s", python_exec, python_implementation)
        LOGGER.info("%s version is: %s", python_exec, subprocess.check_output(
            [python_exec, "--version"], stderr=subprocess.STDOUT, universal_newlines=True).strip())
        if opts.testnames is None:
            for test_goal in python_test_goals:
                task_queue.put((100, (python_exec, test_goal)))
        else:
            for test_goal in testnames_to_test:
                task_queue.put((0, (python_exec, test_goal)))

    # Create the target directory before starting tasks to avoid races.
    target_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'target'))
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)

    def process_queue(task_queue):
        while True:
            try:
                (priority, (python_exec, test_goal)) = task_queue.get_nowait()
            except Queue.Empty:
                break
            try:
                run_individual_python_test(target_dir, test_goal, python_exec, params)
            finally:
                task_queue.task_done()

    start_time = time.time()
    for _ in range(opts.parallelism):
        worker = Thread(target=process_queue, args=(task_queue,))
        worker.daemon = True
        worker.start()
    try:
        task_queue.join()
    except (KeyboardInterrupt, SystemExit):
        print_red("Exiting due to interrupt")
        sys.exit(-1)
    total_duration = time.time() - start_time
    LOGGER.info("Tests passed in %i seconds", total_duration)

    for key, lines in sorted(SKIPPED_TESTS.items()):
        pyspark_python, test_name = key
        LOGGER.info("\nSkipped tests in %s with %s:" % (test_name, pyspark_python))
        for line in lines:
            LOGGER.info("    %s" % line.rstrip())


if __name__ == "__main__":
    SKIPPED_TESTS = Manager().dict()
    main()
