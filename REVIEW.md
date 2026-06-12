# Review instructions

Please do an analysis of the code. If not instructed otherwise look at the diff
from the last commit (git diff HEAD~1) and analyze the changes. It's also possible
I ask you to look at the branch diff to main (git diff main..HEAD).

The root of where we work isn't a git repo, you have info about this in CLAUDE.md
but basically you need to cd into the two dirs and do git commands there.

- I want to make sure we do not have any critical showstoppers that cause crashes
  or unexpected behavior.
- Also check if we have duplicated code and if there is any code that really
  should be cleaned up.
- Also check if there are potential performance issues with the code that really
  should be optimized.
- Everything works as expected as far as I have tested.
- Don't look for style issue.
- Do change the code as you see fit but for major things we need to discuss it first.
