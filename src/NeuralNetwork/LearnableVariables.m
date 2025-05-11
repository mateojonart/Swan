classdef LearnableVariables < handle

    properties (Access = public)
        thetavec
    end

    properties (Access = private)
        neuronsPerLayer
        nLayers
    end

    methods (Access = public)

        function obj = LearnableVariables(s)
            obj.init(s);
            obj.computeInitialTheta()
        end

        function [W,b] = reshapeInLayerForm(obj)
            theta = obj.thetavec;
            nPL = obj.neuronsPerLayer;
            last = 1;
            b = cell(obj.nLayers-1,1);
            W = cell(obj.nLayers-1,1);
            for i = 2:obj.nLayers
                aux = nPL(i)*nPL(i-1) + nPL(i);
                next = last + aux;
                thetaI = theta(last:next-1);

                prevL = nPL(i-1);
                nextL = nPL(i);

                b{i-1} = obj.getB(thetaI,prevL,nextL);
                W{i-1} = obj.getW(thetaI,prevL,nextL);
                last = next;
            end
        end
    end

    methods (Access = private)

        function init(obj,s)
            obj.neuronsPerLayer = s.neuronsPerLayer;
            obj.nLayers         = s.nLayers;
        end

        function computeInitialTheta(obj)
            nPL = obj.neuronsPerLayer;
            tot = sum(nPL(1:end-1) .* nPL(2:end)) + sum(nPL(2:end));
            th = zeros(1, tot);
            pos = 1;
            for i = 2:obj.nLayers
                nPrev = nPL(i-1);
                nCurrent = nPL(i);
                u = sqrt(6 / (nPrev + nCurrent));
                W = unifrnd(-u, u, [1, nPrev * nCurrent]);
                if i ~= obj.nLayers
                    b = zeros(1, nCurrent) + 0.1;
                else
                    b = zeros(1, nCurrent) + 1/nCurrent;
                end

                th(pos : pos + numel(W) + numel(b) - 1) = [W, b];
                pos = pos + numel(W) + numel(b);
            end
            obj.thetavec = th;
        end
    end

    methods (Access = private, Static)

        function value = getW(thv,prevL,nextL)
            aux   = thv(1:prevL*nextL);
            value = reshape(aux,[prevL,nextL]);
        end

        function value = getB(thv,prevL,nextL)
            value = thv(prevL*nextL+1:end);
        end


    end
end

