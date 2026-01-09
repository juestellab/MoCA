using GLMakie

function interactive_figure(E, log_dynamic_range=6)
    f = Figure()
    ax1 = Axis(f[1,1], aspect=DataAspect())
    ax2 = Axis(f[1,2], aspect=DataAspect())
    sl = Slider(f[2,1:2], range=1:size(E, 2), startvalue=size(E, 2)÷2)
    E_slice = lift(sl.value) do slice
        E[:,slice,:]
    end
    heatmap!(ax1, E_slice, colorrange=extrema(E), colormap=:inferno)
    heatmap!(ax2, E_slice, colorscale=log10, colorrange=(maximum(E)*1e1^(-log_dynamic_range), maximum(E)), lowclip=:gray, colormap=:inferno)
    return f
end

