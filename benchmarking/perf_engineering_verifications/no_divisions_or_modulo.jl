using CSV, DataFrames

# How many instructions to scan around a MUFU.RCP to decide whether it's
# part of an FP-based integer division (sandwiched between I2F and F2I)
# or a standalone FP reciprocal.
const WINDOW = 8

opcode(src::AbstractString) = let s = strip(src); isempty(s) ? "" : split(s)[1] end

# Load SASS-page CSV. Row 1 = banner (kernel name), row 2 = column
# header, rows 3+ = per-instruction entries.
function load_sass(path)
    df = CSV.File(path; header=2, skipto=3, types=String,
                  silencewarnings=true) |> DataFrame
    # drop blank trailing rows that CSV sometimes emits
    filter!(r -> !ismissing(r.Source) && !isempty(strip(r.Source)), df)
    return df
end

# Scan an instruction stream and return suspicious indices
#   where indices of MUFU.RCP that are sandwiched by I2F (within WINDOW
#   instructions before) AND F2I (within WINDOW after): the
#   likely FP-based integer division
function find_suspect_indices(ops::Vector{String})
    suspects = Tuple{Int,String}[]
    for (i, op) in enumerate(ops)
        if startswith(op, "MUFU.RCP")
            # look back for I2F and forward for F2I
            lo = max(1, i - WINDOW)
            hi = min(length(ops), i + WINDOW)
            saw_i2f = any(startswith.(ops[lo:i-1], "I2F"))
            saw_f2i = any(startswith.(ops[i+1:hi], "F2I"))
            if saw_i2f && saw_f2i
                push!(suspects, (i, "MUFU.RCP sandwiched by I2F/F2I (FP-based int div)"))
            end
        end
    end
    return suspects
end

# Summary counts the script prints for context.
function counts(ops::Vector{String})
    n_imad_hi = count(startswith.(ops, "IMAD.HI") .| startswith.(ops, "IMUL.HI"))
    n_shf     = count(startswith.(ops, "SHF.R"))
    n_mufu    = count(startswith.(ops, "MUFU"))
    n_i2f     = count(startswith.(ops, "I2F"))
    n_f2i     = count(startswith.(ops, "F2I"))
    n_total   = length(ops)
    return (; n_total, n_imad_hi, n_shf, n_mufu, n_i2f, n_f2i)
end

function check_one(D::Int, op::String)
    profile_dir = joinpath(@__DIR__, "..", "roofline", op,
                             "profile_results", "tuned")
    path = joinpath(profile_dir, "$(op)_ours_D$(D)_sass.csv")
    isfile(path) || (println("  D=$D: FILE NOT FOUND ($path)"); return false)

    df = load_sass(path)
    ops = String[opcode(s) for s in df.Source]
    c = counts(ops)
    sus = find_suspect_indices(ops)

    status = isempty(sus) ? "OK" : "SUSPECT"
    println("  D=$(lpad(D,3))  $(rpad(status,8))  " *
            "total=$(lpad(c.n_total,7))  IMAD.HI=$(lpad(c.n_imad_hi,5))  " *
            "SHF.R=$(lpad(c.n_shf,5))  MUFU=$(c.n_mufu)  " *
            "I2F=$(c.n_i2f)  F2I=$(c.n_f2i)")

    if !isempty(sus)
        for (i, why) in sus
            ctx_lo = max(1, i-2); ctx_hi = min(length(ops), i+2)
            ctx = join(ops[ctx_lo:ctx_hi], " | ")
            println("        line $i: $why")
            println("        context: $ctx")
        end
    end
    return isempty(sus)
end

function check_div(Ds, op)
    println("=== runtime integer-division check on kalman_ours SASS ===")
    println("    Operation: $op")
    println()
    println("    legend:")
    println("      IMAD.HI / SHF.R   expected (magic-number int division)")
    println("      MUFU              MUFU.RCP/RSQ -- only flagged if I2F/F2I-sandwiched")
    println("      I2F / F2I         legitimate int<->float conversions on their own")
    println()

    all_ok = true
    for D in Ds
        all_ok &= check_one(D, op)
    end

    println()
    if all_ok
        println("RESULT: no runtime integer division detected in any kernel.")
    else
        println("RESULT: suspect instructions found -- inspect above output.")
    end
end

Ds = collect(2:32)
for op in ["kalman", "sqrt_kalman", "gauss_likelihood"]
    check_div(Ds, op)
end

#=
=== runtime integer-division check on kalman_ours SASS ===
    Operation: kalman

    legend:
      IMAD.HI / SHF.R   expected (magic-number int division)
      MUFU              MUFU.RCP/RSQ -- only flagged if I2F/F2I-sandwiched
      I2F / F2I         legitimate int<->float conversions on their own

  D=  2  OK        total=   1104  IMAD.HI=    0  SHF.R=   52  MUFU=9  I2F=0  F2I=0
  D=  3  OK        total=   1352  IMAD.HI=    8  SHF.R=   50  MUFU=13  I2F=0  F2I=0
  D=  4  OK        total=   1608  IMAD.HI=    0  SHF.R=   53  MUFU=17  I2F=0  F2I=0
  D=  5  OK        total=   1992  IMAD.HI=   12  SHF.R=   65  MUFU=21  I2F=0  F2I=0
  D=  6  OK        total=   2352  IMAD.HI=   15  SHF.R=   55  MUFU=25  I2F=0  F2I=0
  D=  7  OK        total=   2792  IMAD.HI=   16  SHF.R=   72  MUFU=29  I2F=0  F2I=0
  D=  8  OK        total=   3240  IMAD.HI=    0  SHF.R=   80  MUFU=33  I2F=0  F2I=0
  D=  9  OK        total=   3632  IMAD.HI=   20  SHF.R=   62  MUFU=37  I2F=0  F2I=0
  D= 10  OK        total=   4184  IMAD.HI=   23  SHF.R=   74  MUFU=41  I2F=0  F2I=0
  D= 11  OK        total=   4648  IMAD.HI=   24  SHF.R=   66  MUFU=45  I2F=0  F2I=0
  D= 12  OK        total=   5312  IMAD.HI=   40  SHF.R=   80  MUFU=49  I2F=0  F2I=0
  D= 13  OK        total=   5912  IMAD.HI=   28  SHF.R=   70  MUFU=53  I2F=0  F2I=0
  D= 14  OK        total=   6680  IMAD.HI=   31  SHF.R=   87  MUFU=57  I2F=0  F2I=0
  D= 15  OK        total=   7424  IMAD.HI=   32  SHF.R=   74  MUFU=61  I2F=0  F2I=0
  D= 16  OK        total=   8216  IMAD.HI=    0  SHF.R=   96  MUFU=65  I2F=0  F2I=0
  D= 17  OK        total=   8720  IMAD.HI=   36  SHF.R=   78  MUFU=69  I2F=0  F2I=0
  D= 18  OK        total=   9320  IMAD.HI=   76  SHF.R=  108  MUFU=73  I2F=0  F2I=0
  D= 19  OK        total=  10376  IMAD.HI=   65  SHF.R=   96  MUFU=77  I2F=0  F2I=0
  D= 20  OK        total=  11008  IMAD.HI=   85  SHF.R=  115  MUFU=81  I2F=0  F2I=0
  D= 21  OK        total=  12160  IMAD.HI=   71  SHF.R=  100  MUFU=85  I2F=0  F2I=0
  D= 22  OK        total=  13384  IMAD.HI=   98  SHF.R=  140  MUFU=89  I2F=0  F2I=0
  D= 23  OK        total=  13896  IMAD.HI=   73  SHF.R=   99  MUFU=93  I2F=0  F2I=0
  D= 24  OK        total=  15536  IMAD.HI=  134  SHF.R=  159  MUFU=97  I2F=0  F2I=0
  D= 25  OK        total=  16608  IMAD.HI=  109  SHF.R=  132  MUFU=101  I2F=0  F2I=0
  D= 26  OK        total=  17968  IMAD.HI=  140  SHF.R=  161  MUFU=105  I2F=0  F2I=0
  D= 27  OK        total=  18344  IMAD.HI=   87  SHF.R=  107  MUFU=109  I2F=0  F2I=0
  D= 28  OK        total=  20256  IMAD.HI=  122  SHF.R=  140  MUFU=113  I2F=0  F2I=0
  D= 29  OK        total=  20952  IMAD.HI=   95  SHF.R=  111  MUFU=117  I2F=0  F2I=0
  D= 30  OK        total=  22808  IMAD.HI=  130  SHF.R=  144  MUFU=121  I2F=0  F2I=0
  D= 31  OK        total=  23536  IMAD.HI=  103  SHF.R=  115  MUFU=125  I2F=0  F2I=0
  D= 32  OK        total=  24328  IMAD.HI=    0  SHF.R=  114  MUFU=129  I2F=0  F2I=0

RESULT: no runtime integer division detected in any kernel.
=== runtime integer-division check on kalman_ours SASS ===
    Operation: sqrt_kalman

    legend:
      IMAD.HI / SHF.R   expected (magic-number int division)
      MUFU              MUFU.RCP/RSQ -- only flagged if I2F/F2I-sandwiched
      I2F / F2I         legitimate int<->float conversions on their own

  D=  2  OK        total=   1744  IMAD.HI=    0  SHF.R=   52  MUFU=21  I2F=0  F2I=0
  D=  3  OK        total=   2952  IMAD.HI=    8  SHF.R=   50  MUFU=32  I2F=0  F2I=0
  D=  4  OK        total=   3960  IMAD.HI=    0  SHF.R=   53  MUFU=43  I2F=0  F2I=0
  D=  5  OK        total=   6032  IMAD.HI=   12  SHF.R=   65  MUFU=54  I2F=0  F2I=0
  D=  6  OK        total=   7952  IMAD.HI=   15  SHF.R=   55  MUFU=65  I2F=0  F2I=0
  D=  7  OK        total=  10064  IMAD.HI=   16  SHF.R=   72  MUFU=76  I2F=0  F2I=0
  D=  8  OK        total=  12328  IMAD.HI=    0  SHF.R=   77  MUFU=87  I2F=0  F2I=0
  D=  9  OK        total=  16952  IMAD.HI=   20  SHF.R=   62  MUFU=98  I2F=0  F2I=0
  D= 10  OK        total=  20384  IMAD.HI=   23  SHF.R=   74  MUFU=109  I2F=0  F2I=0
  D= 11  OK        total=  23632  IMAD.HI=   24  SHF.R=   66  MUFU=120  I2F=0  F2I=0
  D= 12  OK        total=  27792  IMAD.HI=   52  SHF.R=   92  MUFU=131  I2F=0  F2I=0
  D= 13  OK        total=  31464  IMAD.HI=   28  SHF.R=   70  MUFU=142  I2F=0  F2I=0
  D= 14  OK        total=  33840  IMAD.HI=   31  SHF.R=   87  MUFU=153  I2F=0  F2I=0
  D= 15  OK        total=  40000  IMAD.HI=   32  SHF.R=   74  MUFU=164  I2F=0  F2I=0
  D= 16  OK        total=  45712  IMAD.HI=    0  SHF.R=   88  MUFU=175  I2F=0  F2I=0
  D= 17  OK        total=  56080  IMAD.HI=   36  SHF.R=   78  MUFU=186  I2F=0  F2I=0
  D= 18  OK        total=  58952  IMAD.HI=   93  SHF.R=  125  MUFU=197  I2F=0  F2I=0
  D= 19  OK        total=  65064  IMAD.HI=   65  SHF.R=   96  MUFU=208  I2F=0  F2I=0
  D= 20  OK        total=  71136  IMAD.HI=  103  SHF.R=  133  MUFU=219  I2F=0  F2I=0
  D= 21  OK        total=  77920  IMAD.HI=   71  SHF.R=  100  MUFU=230  I2F=0  F2I=0
  D= 22  OK        total=  85136  IMAD.HI=  119  SHF.R=  161  MUFU=241  I2F=0  F2I=0
  D= 23  OK        total=  91424  IMAD.HI=   73  SHF.R=   99  MUFU=252  I2F=0  F2I=0
  D= 24  OK        total=  99664  IMAD.HI=  153  SHF.R=  178  MUFU=263  I2F=0  F2I=0
  D= 25  OK        total= 107056  IMAD.HI=  109  SHF.R=  132  MUFU=274  I2F=0  F2I=0
  D= 26  OK        total= 115408  IMAD.HI=  165  SHF.R=  186  MUFU=285  I2F=0  F2I=0
  D= 27  OK        total= 122624  IMAD.HI=   87  SHF.R=  107  MUFU=296  I2F=0  F2I=0
  D= 28  OK        total= 131528  IMAD.HI=  147  SHF.R=  165  MUFU=307  I2F=0  F2I=0
  D= 29  OK        total= 140000  IMAD.HI=   95  SHF.R=  111  MUFU=318  I2F=0  F2I=0
  D= 30  OK        total= 149296  IMAD.HI=  159  SHF.R=  173  MUFU=329  I2F=0  F2I=0
  D= 31  OK        total= 158160  IMAD.HI=  103  SHF.R=  115  MUFU=340  I2F=0  F2I=0
  D= 32  OK        total= 179904  IMAD.HI=    0  SHF.R=   83  MUFU=351  I2F=0  F2I=0

RESULT: no runtime integer division detected in any kernel.
=== runtime integer-division check on kalman_ours SASS ===
    Operation: gauss_likelihood

    legend:
      IMAD.HI / SHF.R   expected (magic-number int division)
      MUFU              MUFU.RCP/RSQ -- only flagged if I2F/F2I-sandwiched
      I2F / F2I         legitimate int<->float conversions on their own

  D=  2  OK        total=   1096  IMAD.HI=    0  SHF.R=   62  MUFU=7  I2F=1  F2I=0
  D=  3  OK        total=   1376  IMAD.HI=    5  SHF.R=   72  MUFU=10  I2F=1  F2I=0
  D=  4  OK        total=   1400  IMAD.HI=    0  SHF.R=   66  MUFU=13  I2F=1  F2I=0
  D=  5  OK        total=   1768  IMAD.HI=    7  SHF.R=   86  MUFU=16  I2F=1  F2I=0
  D=  6  OK        total=   1880  IMAD.HI=    8  SHF.R=   75  MUFU=19  I2F=1  F2I=0
  D=  7  OK        total=   2152  IMAD.HI=    9  SHF.R=   94  MUFU=22  I2F=1  F2I=0
  D=  8  OK        total=   2120  IMAD.HI=    0  SHF.R=   74  MUFU=25  I2F=1  F2I=0
  D=  9  OK        total=   2440  IMAD.HI=   11  SHF.R=   79  MUFU=28  I2F=1  F2I=0
  D= 10  OK        total=   2696  IMAD.HI=   12  SHF.R=   91  MUFU=31  I2F=1  F2I=0
  D= 11  OK        total=   2856  IMAD.HI=   13  SHF.R=   83  MUFU=34  I2F=1  F2I=0
  D= 12  OK        total=   3008  IMAD.HI=   14  SHF.R=   84  MUFU=37  I2F=1  F2I=0
  D= 13  OK        total=   3320  IMAD.HI=   15  SHF.R=   85  MUFU=40  I2F=1  F2I=0
  D= 14  OK        total=   3632  IMAD.HI=   16  SHF.R=  101  MUFU=43  I2F=1  F2I=0
  D= 15  OK        total=   3856  IMAD.HI=   17  SHF.R=   86  MUFU=46  I2F=1  F2I=0
  D= 16  OK        total=   3896  IMAD.HI=    0  SHF.R=   77  MUFU=49  I2F=1  F2I=0
  D= 17  OK        total=   4272  IMAD.HI=   19  SHF.R=   87  MUFU=52  I2F=1  F2I=0
  D= 18  OK        total=   4504  IMAD.HI=   20  SHF.R=   87  MUFU=55  I2F=1  F2I=0
  D= 19  OK        total=   4928  IMAD.HI=   21  SHF.R=  112  MUFU=58  I2F=1  F2I=0
  D= 20  OK        total=   5048  IMAD.HI=   22  SHF.R=  103  MUFU=61  I2F=1  F2I=0
  D= 21  OK        total=   5400  IMAD.HI=   23  SHF.R=   90  MUFU=64  I2F=1  F2I=0
  D= 22  OK        total=   5784  IMAD.HI=   24  SHF.R=   91  MUFU=67  I2F=1  F2I=0
  D= 23  OK        total=   6088  IMAD.HI=   25  SHF.R=   92  MUFU=70  I2F=1  F2I=0
  D= 24  OK        total=   6288  IMAD.HI=   26  SHF.R=   94  MUFU=73  I2F=1  F2I=0
  D= 25  OK        total=   6904  IMAD.HI=   27  SHF.R=  135  MUFU=76  I2F=1  F2I=0
  D= 26  OK        total=   7184  IMAD.HI=   28  SHF.R=   96  MUFU=79  I2F=1  F2I=0
  D= 27  OK        total=   7608  IMAD.HI=   29  SHF.R=  142  MUFU=82  I2F=1  F2I=0
  D= 28  OK        total=   7816  IMAD.HI=   30  SHF.R=  123  MUFU=85  I2F=1  F2I=0
  D= 29  OK        total=   8440  IMAD.HI=   31  SHF.R=  153  MUFU=88  I2F=1  F2I=0
  D= 30  OK        total=   8568  IMAD.HI=   32  SHF.R=   99  MUFU=91  I2F=1  F2I=0
  D= 31  OK        total=   9376  IMAD.HI=   33  SHF.R=  163  MUFU=94  I2F=1  F2I=0
  D= 32  OK        total=   8888  IMAD.HI=    0  SHF.R=  149  MUFU=97  I2F=1  F2I=0

RESULT: no runtime integer division detected in any kernel.
=#