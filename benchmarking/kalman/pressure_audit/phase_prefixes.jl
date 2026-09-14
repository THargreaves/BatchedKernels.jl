include("ph_inputs.jl")
function prefix(P,H,A,Q,R,::Val{S}) where S
    ap=A*P
    S==1 && return ap
    predicted=ap*A'+Q
    S==2 && return predicted
    rhs=H*predicted
    S==3 && return rhs
    innovation=rhs*H'+R
    S==4 && return innovation
    U=cholesky(innovation).U
    S==5 && return U
    forward=U'\rhs
    S==6 && return forward
    gain=U\forward
    S==7 && return gain
    correction=I-gain'*H
    S==8 && return correction
    return correction*predicted
end
function prefix_reference(P,H,A,Q,R,s)
    ap=A*P; s==1 && return ap
    predicted=ap*A'+Q; s==2 && return predicted
    rhs=H*predicted; s==3 && return rhs
    innovation=rhs*H'+R; s==4 && return innovation
    U=Matrix(cholesky(Symmetric(innovation,:U)).U); s==5 && return U
    forward=LowerTriangular(U')\rhs; s==6 && return forward
    gain=UpperTriangular(U)\forward; s==7 && return gain
    correction=I-gain'*H; s==8 && return correction
    return correction*predicted
end
function prefix_assignment(t,policy)
    a=forced_assignment(t,:register;nthreads=128)
    products=Int[]; sums=Int[]
    for (id,node) in enumerate(t.nodes)
        node isa BK.CallNode || continue
        if BK._isplacement(node.fn)
            input=t.nodes[only(t.nodes[only(node.args).id].args).id]
            if input.index==2
                a.orientations[id]=:col
                if policy==:shared_H_predicted
                    a.residences[id]=:single
                    a.orientations[only(node.args).id]=:col
                end
            end
        elseif node.fn==(*)
            push!(products,id)
            if length(products)>=5
                a.orientations[id]=:col; a.variants[id]=:matmul_col
            end
        elseif node.fn==(+)
            push!(sums,id)
            if length(sums)==1 && policy==:shared_H_predicted
                a.residences[id]=:single
            end
        elseif BK._isstage(node.fn)
            # Match producer orientation to avoid a gratuitous transpose.
            a.orientations[id]=length(products)>=5 ? :col : :row
        end
    end
    raw=[i for (i,node) in enumerate(t.nodes) if !(node isa BK.NewNode)]
    return BK.Assignment(t;residences=Dict(i=>a.residences[i] for i in raw if haskey(a.residences,i)),orientations=Dict(i=>a.orientations[i] for i in raw if haskey(a.orientations,i)),variants=a.variants,nthreads=128)
end
if get(ENV,"PREFIX_DEFINITIONS_ONLY","false") != "true"
for s in parse.(Int,split(get(ENV,"STAGES","1,2,3,4,5,6,7,8,9"),","))
    f=@eval $(Symbol(:prefix_,s))(P,H,A,Q,R)=prefix(P,H,A,Q,R,Val($s))
    f=getfield(Main,Symbol(:prefix_,s))
    t=BK.trace(f,BK.InputSpec[BK.input_spec(x) for x in args])
    ref=similar(P)
    for n=1:N
        ref[:,:,n]=prefix_reference(P[:,:,n],H[:,:,n],A,Q,R,s)
    end
    for policy in (:register,:shared_H_predicted)
        a=prefix_assignment(t,policy)
        prepare(f,args,policy,ref;assignment_override=a)
    end
end
@assert all(Array(x.data)==original for (x,original) in zip(args,inputs))

end
