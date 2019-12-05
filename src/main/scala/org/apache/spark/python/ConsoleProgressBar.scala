/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.python

import java.util.concurrent.atomic.AtomicInteger
import java.util.{Timer, TimerTask}

class ConsoleProgressBar(numTasks: AtomicInteger) {

  private val maxNumTasks = numTasks.get

  // Carriage return
  private val CR = '\r'
  // Update period of progress bar, in milliseconds
  private val updatePeriodMSec = 200L
  // Delay to show up a progress bar, in milliseconds
  private val firstDelayMSec = 500L

  // The width of terminal
  private val TerminalWidth = sys.env.getOrElse("COLUMNS", "80").toInt
  private var lastProgressBar = ""

  // Schedule a refresh thread to run periodically
  private val timer = new Timer("refresh progress", true)
  timer.schedule(new TimerTask{
    override def run(): Unit = {
      refresh()
    }
  }, firstDelayMSec, updatePeriodMSec)

  private def refresh(): Unit = {
    val finishedNumTasks = maxNumTasks - numTasks.get()
    val bar = {
      val header = s"[#Tasks:"
      val tailer = s" $finishedNumTasks/$maxNumTasks]"
      val w = TerminalWidth - header.length - tailer.length
      val bar = if (w > 0) {
        val percent = w * finishedNumTasks / maxNumTasks
        (0 until w).map { i =>
          if (i < percent) "=" else if (i == percent) ">" else " "
        }.mkString("")
      } else {
        ""
      }
      header + bar + tailer
    }
    if (bar != lastProgressBar) {
      System.err.println(CR + bar)
    }
    lastProgressBar = bar
  }

  def stop(): Unit = {
    timer.cancel()
  }
}
