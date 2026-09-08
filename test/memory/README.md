Hybrid storage checks use selected shapes and failure modes, rather than a Cartesian product. Run the CPU and GPU test items with diagnostics off and on in separate Julia processes:

```sh
julia --project=. --startup-file=no -e 'using TestEnv; TestEnv.activate(); using Preferences; Preferences.set_preferences!(Base.UUID("2fc592bb-64fa-475c-8993-177b89068dc6"), "debug_accessors"=>false; force=true); using TestItemRunner; TestItemRunner.run_tests("test/memory"; verbose=true)'
```

Repeat with `true`. Preferences are written into TestEnv's temporary environment. Zero-local-memory gates apply to production mode; debug assertions introduce exception paths and change compiled resources. The negative resource fixture is compiled and inspected, not launched.

Run the deliberate nonuniform broadcast separately because device assertions can invalidate the CUDA context:

```sh
julia -g2 --project=. --startup-file=no -e 'using TestEnv; TestEnv.activate(); using Preferences; Preferences.set_preferences!(Base.UUID("2fc592bb-64fa-475c-8993-177b89068dc6"), "debug_accessors"=>true; force=true); include("test/memory/debug_violation.jl")'
```

This command succeeds only when execution raises the expected CUDA kernel exception. Its device error output should identify the group-uniformity assertion. Compilation occurs outside the expected-exception check.
