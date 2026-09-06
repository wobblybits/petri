import { Sim } from './sim.ts';

/**
 * A quarter-size dish for the correctness projects.
 *
 * The CPU field is a fixed cost per frame — two diffusions, a decay and a
 * growth over every cell of the disk — and at the world's full size that is
 * twenty milliseconds a frame in Node, before a single body is touched. A
 * test that steps a small net for ten seconds paid twelve of those on the
 * field, and seven hundred such tests were most of a suite that took half
 * an hour on a warm machine.
 *
 * 256 cells a side is a sixteenth of the work. The cell is the same ten
 * units, so the sensor spacing, the dead zone and the deposit normalisation
 * all mean what they meant; what changes is that the dish is 2,560 units
 * across instead of 10,240. Every net the suite builds sits in the middle
 * of it with room to spare. The perf and experiment projects do not load
 * this file and run the full dish, since a ledger over a small field would
 * not be a ledger of the thing that ships.
 */
Sim.defaultFieldCells = 256;
