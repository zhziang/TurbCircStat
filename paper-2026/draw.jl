using CairoMakie, HDF5, LinearAlgebra
using FourierFlows, CUDA, Measurements
include("../../src/postproc/circulation.jl")
mytheme = merge(
    Theme(
        Axis=(
            xgridvisible=false,
            xminorticksvisible=true,
            xminorticksize=2,
            xticksize=3,
            xticklabelsize=8,
            xlabelsize=8,
            xlabelpadding=2,
            xtickalign=1,
            xminortickalign=1,
            ygridvisible=false,
            yminorticksvisible=true,
            yminorticksize=2,
            yticksize=3,
            yticklabelsize=8,
            ylabelsize=8,
            ylabelpadding=2,
            ytickalign=1,
            yminortickalign=1,
        ),
        Lines=(
            joinstyle=:round,
            linewidth=1,
        ),
        Scatter=(
            markersize=5,
        ),
        Legend=(
            labelsize=8,
            padding=6,
            patchsize=(10, 5),
            colgap=7,
            patchlabelgap=3,
            framevisible=false,
        ),
        Errorbars=(
            linewidth=0.5,
            whiskerwidth=2,
        ),
        Text=(
            fontsize=8,
        ),
    ),
    theme_latexfonts(),
)

## Global constants
pfile = h5open((@__DIR__) * "/postdata.h5", "r")
cm = 96 / 2.54
pt = 4 / 3
Reynolds = [3.5; 3.7; 3.9; 4.0; 4.02; 4.04; 4.05]
αβs = map(Reynolds) do Re
    data = pfile["wavenumber-energy_spectra"]["$(Re)"]
    x, y = read(data["x"]), read(data["y"])
    αβ = hcat(y[1:300], x[1:300] .^ 2 .* y[1:300]) \ (2π .* x[1:300])
    Re => (αβ[1], αβ[2])
end
αβs = Dict(αβs)

colormaps = cgrad(:roma, length(Reynolds); categorical=true)
eqSpec(x, Re) = @. 2π * x / (αβs[Re][1] + αβs[Re][2] * x^2)

## Figure 1
with_theme(mytheme) do
    fig = Figure(size=(8.5cm, 8.2cm))

    # Layout
    begin
        ga = GridLayout(fig[1, 1])
        axa = Axis(ga[1, 1];
            xlabel=L"k",
            ylabel=L"E(k)",
            xscale=log10,
            yscale=log10,
            limits=(1, 512, 1e-6, 1),
            xticks=2 .^ (0:2:8),
            yticks=10.0 .^ (-6:-1),
            xminorticks=IntervalsBetween(3),
            yminorticks=IntervalsBetween(9),
            ytickformat=values -> [L"10^{%$(round(Int, log10(v)))}" for v in values],
        )

        gb = GridLayout(fig[1, 2])
        axb = Axis(gb[1, 1];
            xlabel=L"u/\sqrt{⟨u^2⟩}",
            ylabel="PDF",
            yscale=log10,
            limits=(-7, 7, 1e-10, 1),
            yticks=10.0 .^ (-10:2:-2),
            yminorticks=10.0 .^ (-10:-1),
            ytickformat=values -> [L"10^{%$(round(Int, log10(v)))}" for v in values],
        )



        gc = GridLayout(fig[2, 1])
        axct = Axis(gc[1, 1];
            ylabel=L"\Pi_E/ε",
            xscale=log10,
            limits=(1, 512, -0.6, 0.4),
            xticklabelsvisible=false,
            xticks=2 .^ (0:2:8),
            yticks=-0.6:0.2:0.2,
            yticklabelpad=2,
            xminorticks=IntervalsBetween(3),
        )

        axct_in = Axis(gc[1, 1];
            height=Relative(0.4),
            width=Relative(0.6),
            halign=0.1,
            valign=0.15,
            xscale=log10,
            xticklabelsvisible=false,
            yticklabelsvisible=false,
            xticksvisible=false,
            xminorticksvisible=false,
            yticksvisible=false,
            yminorticksvisible=false,
            spinewidth=0.7,
            limits=(2, 32, -1e-2, 5e-3),
            xminorticks=IntervalsBetween(3),
            ytickformat=values -> [L"%$(round(Int, 100*v))%" for v in values],
        )
        axcb = Axis(gc[2, 1];
            xlabel=L"k",
            ylabel=L"\Pi_\Omega/η",
            xscale=log10,
            limits=(1, 512, -0.3, 0.6),
            xticks=2 .^ (0:2:8),
            yticks=-0.2:0.2:0.4,
            yticklabelpad=2,
            xminorticks=IntervalsBetween(3),
        )
        axcb_in = Axis(gc[2, 1];
            height=Relative(0.4),
            width=Relative(0.6),
            halign=0.1,
            valign=0.75,
            xscale=log10,
            xticklabelsvisible=false,
            yticklabelsvisible=false,
            xticksvisible=false,
            xminorticksvisible=false,
            yticksvisible=false,
            yminorticksvisible=false,
            spinewidth=0.7,
            limits=(2, 32, -1e-4, 5e-5),
            xminorticks=IntervalsBetween(3),
            ytickformat=values -> [L"%$(round(Int, 100*v))%" for v in values],
        )
        rowgap!(gc, 8)

        gd = GridLayout(fig[2, 2])
        axd = Axis(gd[1, 1];
            xlabel=L"Re",
            ylabel=L"l_\text{eq}k_f",
            yscale=log10,
            limits=(3.2, 4.1, 1, 10^3),
            yticks=10 .^ (0:2),
            yminorticks=IntervalsBetween(9),
            ytickformat=values -> [L"10^{%$(round(Int, log10(v)))}" for v in values],
        )
        axd_in = Axis(gd[1, 1];
            height=Relative(0.4),
            width=Relative(0.4),
            halign=0.4,
            valign=0.8,
            xlabel=L"Re",
            ylabel=L"⟨u^4⟩/⟨u^2⟩^2",
            limits=(3.2, 4.2, 1, 4),
            xticks=3.2:0.5:4.2,
            xminorticks=IntervalsBetween(5),
            yticklabelpad=1,
            xlabelpadding=1,
        )

        for (gl, lb) in zip([ga, gb, gc, gd], ["(a)", "(b)", "(c)", "(d)"])
            Label(gl[1, 1, TopLeft()], lb;
                fontsize=10,
                font=:bold,
                padding=(0, 10, -10, 0),
            )
        end

        rowgap!(fig.layout, 10)
        colgap!(fig.layout, 10)
    end

    #data
    begin
        for (Re, c) in zip(Reynolds, colormaps)
            data = pfile["wavenumber-energy_spectra"]["$(Re)"]
            α, β = αβs[Re]
            x, y = read(data["x"]), read(data["y"])
            lines!(axa, x, y; color=c)
            if αβs[Re][1] > 0 && αβs[Re][2] > 0
                xref = x[4:300]
                lines!(axa, xref, eqSpec(xref, Re); linestyle=:dashdot, color=:black)
            end
        end

        for (Re, c) in zip(Reynolds, colormaps)
            data = pfile["velocity-pdf:filtered"]["$(Re)"]
            x, y, Δy = read(data["x"]), read(data["y"]), read(data["Δy"])
            std = sqrt(sum(@. y * x^2) * (x[2] - x[1]))
            kts = sum(@. y * x^4) * (x[2] - x[1])
            scatter!(axb, x ./ std, y .* std;
                strokecolor=c,
                color=:white,
                strokewidth=1,
                markersize=5,
            )
            lines!(axb, x ./ std, y .* std;
                color=c,
                linewidth=0.5,
            )
            scatter!(axd_in, Re, kts / std^4;
                color=:white,
                strokecolor=c,
                strokewidth=1,
                markersize=5,
            )
            errorbars!(axb, x ./ std, y .* std, Δy .* std; color=c)
        end

        for (Re, c) in zip(Reynolds, colormaps)
            efdata, wfdata = pfile["wavenumber-energy_fluxes"]["$(Re)"], pfile["wavenumber-enstrophy_fluxes"]["$(Re)"]
            x1, y1 = read(efdata["x"]), read(efdata["y"])
            x2, y2 = read(wfdata["x"]), read(wfdata["y"])
            lines!(axct, x1, y1; color=c)
            lines!(axct_in, x1, y1; color=c)
            lines!(axcb, x2, y2; color=c)
            lines!(axcb_in, x2, y2; color=c)
        end

        x = [Re for Re in Reynolds if αβs[Re][1] > 0 && αβs[Re][2] > 0]
        y = [sqrt(αβs[Re][2] / αβs[Re][1]) * 1024 / 3sqrt(Re) for Re in Reynolds if αβs[Re][1] > 0 && αβs[Re][2] > 0]
        lines!(axd, x, y; color=:black, alpha=0.2)
        for (Re, c) in zip(Reynolds, colormaps)
            if αβs[Re][1] > 0 && αβs[Re][2] > 0
                scatter!(axd, Re, sqrt(αβs[Re][2] / αβs[Re][1]) * 1024 / 3sqrt(Re); strokecolor=c,
                    color=:white,
                    strokewidth=1)
            end
        end

    end

    # Decorations
    begin
        lines!(axa, 2:64, 2e-6 .* (2:64); linestyle=:dash, color=:red)
        lines!(axa, 2:64, 1e-1 .* (2:64) .^ (-1); linestyle=:dash, color=:blue)
        text!(axa, 16, 1e-5; text=L"k", color=:red)
        text!(axa, 16, 1e-2; text=L"k^{-1}", color=:blue)
        bracket!(axct, 2, 0, 32, 0;
            offset=2,
            style=:square,
            orientation=:up,
            linewidth=0.5,
            width=3,
            text="zoom in",
            fontsize=6,
        )
        bracket!(axcb, 2, 0, 32, 0;
            offset=2,
            style=:square,
            orientation=:down,
            linewidth=0.5,
            width=3,
            text="zoom in",
            fontsize=6,
        )
        bracket!(axct, 0.65, 0.36, 0.65, 0.1;
            space=:relative,
            offset=0,
            style=:square,
            orientation=:up,
            linewidth=0.5,
            width=3,
            text="1%",
            fontsize=6,
        )

        bracket!(axcb, 0.65, 0.72, 0.65, 0.46;
            space=:relative,
            offset=0,
            style=:square,
            orientation=:up,
            linewidth=0.5,
            width=3,
            text="1‱",
            fontsize=6,
        )

        xref = -7:0.05:7
        yref = @. exp(-xref^2 / 2) / sqrt(2π)
        lines!(axb, xref, yref; linestyle=:dashdot, color=:black)


        hlines!(axcb, 0; linestyle=:dashdot, linewidth=1, color=:black)
        hlines!(axct, 0; linestyle=:dashdot, linewidth=1, color=:black)
        hlines!(axcb_in, 0; linestyle=:dashdot, linewidth=1, color=:black)
        hlines!(axct_in, 0; linestyle=:dashdot, linewidth=1, color=:black)
        hlines!(axd_in, 3; linestyle=:dashdot, linewidth=1, color=:black)


        Legend(gb[1, 1],
            [[LineElement(color=c, linestyle=nothing), MarkerElement(marker=:circle, markersize=5, color=:white, strokecolor=c, strokewidth=1)] for c in colormaps],
            ["$(Re)" for Re in Reynolds];
            height=Relative(0.55),
            width=Relative(0.75),
            halign=0.6,
            valign=0.4,
            fontsize=6,
            rowgap=1,
        )
    end


    save((@__DIR__) * "/diag.pdf", fig)
end

## Figure 2
with_theme(mytheme) do
    fig = Figure(size=(8.5cm, 8.2cm))

    # Layout
    begin
        ga = GridLayout(fig[1, 1])
        axa = Axis(ga[1, 1];
            xlabel=L"\sqrt{A_C}/l_\text{eq}",
            ylabel=L"\sqrt{⟨Γ_C^2⟩/⟨Γ_□^2⟩}",
            xscale=log10,
            limits=(1e-2, 1e2, 0.95, 1.2),
            xminorticks=IntervalsBetween(9),
        )

        gb = GridLayout(fig[1, 2])
        axb = Axis(gb[1, 1];
            xlabel=L"l_1/l_2",
            ylabel=L"(⟨Γ_C^p⟩/⟨Γ_□^p⟩)^{1/p}",
            xscale=log10,
            xticks=10.0 .^ (-2:0),
            yticks=0.9:0.05:1.1,
            xminorticks=IntervalsBetween(9),
            yminorticks=IntervalsBetween(5),
            limits=(1e-2, 1, 0.9, 1.08),
        )

        gc = GridLayout(fig[2, 1])
        axc = Axis(gc[1, 1];
            xlabel=L"L_C/4 l_\text{eq}",
            ylabel=L"\sqrt{⟨Γ_C^2⟩/⟨Γ_□^2⟩}",
            xscale=log10,
            limits=(1e-2, 1e2, 0.75, 1.05),
            xminorticks=IntervalsBetween(9),
        )

        gd = GridLayout(fig[2, 2])
        axd = Axis(gd[1, 1];
            xlabel=L"l_1/l_2",
            ylabel=L"(⟨Γ_C^p⟩/⟨Γ_□^p⟩)^{1/p}",
            xscale=log10,
            xticks=10.0 .^ (-2:0),
            yticks=0.9:0.05:1.1,
            xminorticks=IntervalsBetween(9),
            yminorticks=IntervalsBetween(5),
            limits=(1e-2, 1, 0.9, 1.08),
        )

        for (gl, lb) in zip([ga, gb, gc, gd], ["(a)", "(b)", "(c)", "(d)"])
            Label(gl[1, 1, TopLeft()], lb;
                fontsize=10,
                font=:bold,
                padding=(0, 10, -10, 0),
            )
        end

        rowgap!(fig.layout, 10)
        colgap!(fig.layout, 10)
    end

    # Data
    begin
        for (Re, c) in zip(Reynolds, colormaps)
            data = pfile["loop_sizes-square_moments"]["$(Re)"]
            x_rect = read(data["2"]["5×20"]["x"])
            y_rect = read(data["2"]["5×20"]["y"])
            Δy_rect = read(data["2"]["5×20"]["Δy"])
            npoints = length(x_rect)
            x_sq = read(data["2"]["10×10"]["x"])[1:npoints]
            y_sq = read(data["2"]["10×10"]["y"])[1:npoints]
            Δy_sq = read(data["2"]["10×10"]["Δy"])[1:npoints]

            if αβs[Re][1] > 0 && αβs[Re][2] > 0
                leq = sqrt(αβs[Re][2] / αβs[Re][1])
                x, y = 10x_rect ./ 1024leq, sqrt.(y_rect ./ y_sq)
                lines!(axa, x[1:2:(end-3)], y[1:2:(end-3)]; color=c)
                scatter!(axa, x[1:2:(end-3)], y[1:2:(end-3)]; strokecolor=c, color=:white, strokewidth=1)
            end
        end

        for (Re, c) in zip(Reynolds, colormaps)
            data = pfile["loop_sizes-square_moments"]["$(Re)"]
            x_rect = read(data["2"]["4×16"]["x"])
            y_rect = read(data["2"]["4×16"]["y"])
            Δy_rect = read(data["2"]["4×16"]["Δy"])
            npoints = length(x_rect)
            x_sq = read(data["2"]["10×10"]["x"])[1:npoints]
            y_sq = read(data["2"]["10×10"]["y"])[1:npoints]
            Δy_sq = read(data["2"]["10×10"]["Δy"])[1:npoints]

            if αβs[Re][1] > 0 && αβs[Re][2] > 0
                leq = sqrt(αβs[Re][2] / αβs[Re][1])
                x, y = 10x_rect ./ 1024leq, sqrt.(y_rect ./ y_sq)
                lines!(axc, x[1:2:(end-3)], y[1:2:(end-3)]; color=c)
                scatter!(axc, x[1:2:(end-3)], y[1:2:(end-3)]; strokecolor=c, color=:white, strokewidth=1, label="$(Re)")
            end
        end

        markers = [:circle, :utriangle, :dtriangle, :rect, :diamond]
        orders = 2:2:10
        hspan!(axb, 0.99, 1.01; alpha=0.2, color=:black)
        for (n, mk) in zip(orders, markers)
            data = pfile["aspect_ratio-moments"]["area8100"]["4.04"]["$(n)"]
            x = read(data["x"])
            y = read(data["y"])
            Δy = read(data["Δy"])
            scatter!(axb, x, (y ./ y[end]) .^ (1 / n); marker=mk, color=:white, strokecolor=colormaps[6], strokewidth=1, label="$(n)")
        end
        hspan!(axd, 0.99, 1.01; alpha=0.2, color=:black)
        for (n, mk) in zip(orders, markers)
            data = pfile["aspect_ratio-moments"]["perimeter180"]["3.5"]["$(n)"]
            x = read(data["x"])
            y = read(data["y"])
            Δy = read(data["Δy"])
            scatter!(axd, x, (y ./ y[end]) .^ (1 / n); marker=mk, color=:white, strokecolor=colormaps[1], strokewidth=1, label="$(n)")
        end
    end

    # Decorations
    begin
        hlines!(axa, 1; linestyle=:dashdot, linewidth=1, color=:black)
        text!(axa, 0.05, 0.95; text=L"A_C = A_\square", space=:relative, align=(:left, :top))

        hlines!(axc, 1; linestyle=:dashdot, linewidth=1, color=:black)
        axislegend(axc; orientation=:vertical, colgap=2, patchlabelgap=0, valign=0, halign=1, padding=1, nbanks=2)
        text!(axc, 0.05, 0.95; text=L"L_C = L_\square", space=:relative, align=(:left, :top))

        axislegend(axb; orientation=:horizontal, colgap=2, patchlabelgap=0, valign=0, halign=0, padding=0, nbanks=2)
        leq = sqrt(αβs[4.04][2] / αβs[4.04][1])
        text!(axb, 0.5, 0.85; text=L"{A_C} = %$(round((90/1024/leq)^2; digits = 5))\times 4\pi^2 l_\text{eq}^2",
            space=:relative,
            align=(:center, :bottom),
        )
        text!(axb, 0.07, 0.25; text=L"Re = 4.04",
            space=:relative,
            align=(:left, :bottom),
        )
        text!(axb, 0.012, 1; text=L"\pm 1%",
            align=(:left, :center),
        )


        axislegend(axd; orientation=:horizontal, colgap=2, patchlabelgap=0, valign=0, halign=0, padding=0, nbanks=2)
        leq = sqrt(αβs[3.5][2] / αβs[3.5][1])
        text!(axd, 0.5, 0.85; text=L"L_C = %$(round((90/1024/leq); digits = 3))\times 8\pi l_\text{eq}", space=:relative, align=(:center, :bottom))
        text!(axd, 0.07, 0.25; text=L"Re = 3.5",
            space=:relative,
            align=(:left, :bottom),
        )
        text!(axd, 0.012, 1; text=L"\pm 1%",
            align=(:left, :center),
        )
        begin
            grid = TwoDGrid(CPU(); nx=1024, Lx=2π)
            loopsizes = [15; 20; 30; 40; 45; 60; 75; 90]
            rects = [rectsloop(grid, l, 8100 ÷ l, 1, 1) for l in loopsizes]
            aspect_ratios = loopsizes .^ 2 ./ 8100
            ζh_IR = grid.Krsq .^ (-1 / 3)
            CUDA.@allowscalar ζh_IR[1, 1] = 0
            vars = map(rects) do hsh
                Γ = getΓ(ζh_IR, hsh, grid)
                sum(x -> x^2, Γ)
            end
            vars_ratio = vars ./ vars[end]
            lines!(axb, aspect_ratios, sqrt.(vars_ratio); linestyle=:dash, color=:purple)
        end

        begin
            grid = TwoDGrid(CPU(); nx=1024, Lx=2π)
            loopsizes = 10:10:90
            rects = [rectsloop(grid, l, 180 - l, 1, 1) for l in loopsizes]
            aspect_ratios = loopsizes .^ 2 ./ 8100
            ζh_IR = grid.Krsq .^ (-1 / 3)
            CUDA.@allowscalar ζh_IR[1, 1] = 0
            vars = map(rects) do hsh
                Γ = getΓ(ζh_IR, hsh, grid)
                sum(x -> x^2, Γ)
            end
            vars_ratio = vars ./ vars[end]
            lines!(axd, aspect_ratios, sqrt.(vars_ratio); linestyle=:dash, color=:purple)
        end

    end

    save((@__DIR__) * "/areaperi.pdf", fig)
end


## Figure3
with_theme(mytheme) do
    fig = Figure(size=(8.5cm, 8.2cm))

    # Layout
    begin
        ga = GridLayout(fig[1, 1])
        axa = Axis(ga[1, 1];
            xlabel=L"p",
            ylabel=L"\zeta_p",
            limits=(0, 11, 0, 11),
            xticks=1:2:11,
            yticks=1:2:11,
        )

        gb = GridLayout(fig[1, 2])
        axb = Axis(gb[1, 1];
            xlabel=L"l/l_\text{eq}",
            ylabel=L"\alpha^{p/2}⟨|Γ_C|^p⟩l_\text{eq}^{p}l^{-p}",
            xscale=log10,
            yscale=log10,
            xticks=10.0 .^ (-2:2),
            yticks=[1, 1e10, 1e20],
            xminorticks=IntervalsBetween(9),
            yminorticks=10.0 .^ (0:30),
            limits=(5e-3, 1e2, 1e-1, 1e27),
            xtickformat=values -> [L"10^{%$(round(Int, log10(v)))}" for v in values],
            ytickformat=values -> [L"10^{%$(round(Int, log10(v)))}" for v in values],
        )

        gc = GridLayout(fig[2, 1])
        axc = Axis(gc[1, 1];
            xlabel=L"l/l_\text{eq}",
            ylabel=L"α^{p/2}⟨|Γ_C|^p⟩l_\text{eq}^{p/2}l^{-p/2} ",
            xscale=log10,
            yscale=log10,
            xticks=10.0 .^ (-2:2),
            yticks=[1, 1e10, 1e20],
            xminorticks=IntervalsBetween(9),
            yminorticks=10.0 .^ (0:30),
            limits=(5e-3, 1e2, 1e-1, 1e19),
            xtickformat=values -> [L"10^{%$(round(Int, log10(v)))}" for v in values],
            ytickformat=values -> [L"10^{%$(round(Int, log10(v)))}" for v in values],
        )

		refv = [0.7979;1;1.5958;3;6.3831;15;38.2985;105;306.3876;945]
        gd = GridLayout(fig[2, 2])
        axd = Axis(gd[1, 1];
            xlabel=L"l/l_\text{eq}",
            ylabel=L"⟨|Γ_C|^p⟩/⟨Γ_C^2⟩^{p/2}",
            xscale=log10,
            yscale=log10,
            xticks=10.0 .^ (-2:2),
            yticks=refv[2:2:end],
            xminorticks=IntervalsBetween(9),
            yminorticks=[1:3;3:3:15;15:15:105;105:105:945],
            limits=(5e-3, 1e2, 5e-1, 3000),
            xtickformat=values -> [L"10^{%$(round(Int, log10(v)))}" for v in values],
            # ytickformat=values -> [L"10^{%$(round(Int, log10(v)))}" for v in values],
        )

        for (gl, lb) in zip([ga, gb, gc, gd], ["(a)", "(b)", "(c)", "(d)"])
            Label(gl[1, 1, TopLeft()], lb;
                fontsize=10,
                font=:bold,
                padding=(0, 16, -10, 0),
            )
        end

        rowgap!(fig.layout, 10)
        colgap!(fig.layout, 10)
    end

    for (Re, c) in zip(Reynolds, colormaps)
        if αβs[Re][2] > 0 && αβs[Re][1] > 0
            leq = 2π * sqrt(αβs[Re][2] / αβs[Re][1])
            α, β = αβs[Re][1], αβs[Re][2]

            Γ² = read(pfile["loop_sizes-square_moments"]["$(Re)"]["2"]["10×10"]["y"])[1:2:(end-64)]
            for n in 1:10
                data = pfile["loop_sizes-square_moments"]["$(Re)"]["$(n)"]["10×10"]
                l, Γⁿ = (20π / 1024) .* read(data["x"])[1:2:(end-64)], read(data["y"])[1:2:(end-64)]
                x = l ./ leq
                y = Γⁿ .* α^(n / 2)
                y_ref = (Γ² .* α) .^ (n / 2)

                lines!(axb, x, y .* x .^ (-n); color=c)
                scatter!(axb, x, y .* x .^ (-n); color=:white, strokecolor=c, strokewidth=0.5, markersize=3, label="$(Re)")

                lines!(axc, x, y .* x .^ (-n / 2); color=c)
                scatter!(axc, x, y .* x .^ (-n / 2); color=:white, strokecolor=c, strokewidth=0.5, markersize=3, label="$(Re)")

                lines!(axd, x, y ./ y_ref; color=c)
                scatter!(axd, x, y ./ y_ref; color=:white, strokecolor=c, strokewidth=0.5, markersize=3, label="$(Re)")
            end
        end
    end

    axislegend(axb; unique=true, orientation=:vertical, colgap=0, patchlabelgap=0, valign=1, halign=1, padding=0, nbanks=2)

    sc_area = map(1:10) do n
        Re = 4.04
		leq = 2π * sqrt(αβs[Re][2] / αβs[Re][1])
		α, β = αβs[Re][1], αβs[Re][2]
		data = pfile["loop_sizes-square_moments"]["$(Re)"]["$(n)"]["10×10"]
        l, Γⁿ = (20π / 1024) .* read(data["x"])[1:12], read(data["y"])[1:12]
        x = l ./ leq
        y = Γⁿ .* α^(n / 2)
        fit = hcat(log.(x), ones(length(x))) \ log.(y)
        yref = exp.(hcat(log.(x), ones(length(x))) * fit)
        lines!(axb, x, yref .* x .^ (-n); color=:blue)
        fit[1]
    end

    sc_peri = map(1:10) do n
        Re = 3.5
		leq = 2π * sqrt(αβs[Re][2] / αβs[Re][1])
		α, β = αβs[Re][1], αβs[Re][2]
		data = pfile["loop_sizes-square_moments"]["$(Re)"]["$(n)"]["10×10"]
        l, Γⁿ = (20π / 1024) .* read(data["x"])[6:end-32], read(data["y"])[6:end-32]
        x = l ./ leq
        y = Γⁿ .* α^(n / 2)
        fit = hcat(log.(x), ones(length(x))) \ log.(y)
        yref = exp.(hcat(log.(x), ones(length(x))) * fit)
        lines!(axc, x, yref .* x .^ (-n/2); color=:red)
        fit[1]
    end
    scatter!(axa, 1:10, sc_area, strokecolor=:blue, color=:transparent, strokewidth=1, label=L"l\ll 2\pi l_\text{eq},\; Re=4.04")
    scatter!(axa, 1:10, sc_peri, strokecolor=:red, color=:transparent, strokewidth=1, label=L"l\gg 2\pi l_\text{eq},\; Re = 3.5")
    errorbars!(axa, 1:10, Measurements.value.(sc_area), 3 .* Measurements.uncertainty.(sc_area), color=:blue)
    errorbars!(axa, 1:10, Measurements.value.(sc_peri), 3 .* Measurements.uncertainty.(sc_peri), color=:red)
    lines!(axa, 0:11, 0:11; color=:blue, linestyle=:dash)
    lines!(axa, 0:11, 0.5 .* (0:11); color=:red, linestyle=:dash)
    text!(axa, 6.5, 3; text=L"p/2", color=:red, align=(:left, :top))
    text!(axa, 5.5, 6; text=L"p", color=:blue, align=(:right, :bottom))
    
	hlines!(axd, refv; linewidth = 0.5, color = :black, alpha = 0.5)
	axislegend(axa; halign=0, valign=1, padding=0, patchlabelgap = 0)

    save((@__DIR__) * "/scaling.pdf", fig)
end

