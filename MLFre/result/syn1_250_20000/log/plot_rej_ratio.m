function plot_rej_ratio(scale,data_name)

data = load([data_name '_result_log.mat']);
rej_ratio = data.rej_ratio;
Lambda = data.Lambda;

rej_ratio(1:2,end) = 0;

rej_ratio = (flipud(rej_ratio))';
% if Lambda(end)==1
%     Lambda(end)=[];
%     rej_ratio(end,:)=[];
% end

h = figure;
hold on;
set(gca,'FontSize',28)
if strcmp(scale,'linear')
    g=area(Lambda,rej_ratio);
    set(gca,'XTick',[0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8  0.9 1.0])
    set(gca,'XTickLabel',{'0', '0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0'})
else
    set(gca, 'XScale', 'log')
    g=area(Lambda,rej_ratio);
%     set(gca,'XTick',[ 0.01 0.02 0.04 0.1 0.2 0.4 1])
%     set(gca,'XTickLabel',{ '0.01', '0.02', '0.04', '0.1', '0.2', '0.4', '1'})
    set(gca,'XTick',[0.05 0.1 0.2 0.4 1])
    set(gca,'XTickLabel',{ '0.05', '0.1', '0.2', '0.4', '1'})
end

set(gca,'YTick',[ 0.1 0.3 0.5 0.7 0.9 1])
set(gca,'YTickLabel',{ '0.1', '0.3', '0.5', '0.7', '0.9', '1'})

custom_color = [145,191,219;255,255,191;252,141,89]/255;
for i = 1:size(rej_ratio,2)
    set(g(i),'FaceColor',custom_color(i,:));
end
% set(g(1),'FaceColor',[0.3098    0.5059    0.7412]);
% set(g(2),'FaceColor',[0.7529    0.3137    0.3020]);
%set(g(2),'FaceColor',[0.5020    0.3922    0.6353]);
%set(g(1),'FaceColor',[0.6078    0.7333    0.3490]);

method_str_arr = {'Layer 1','Layer 2','Layer 3'};
legend(method_str_arr,'Location','SouthEast')
xlabel({'$$\lambda/\lambda_{\max}$$'},'Interpreter', 'Latex')
ylabel('Rejection Ratio')


%axis([min(Lambda) 1 0 1])
axis([0.05 1 0 1])
box on

% set figure ratio.
pbaspect([1 0.8 1])

figure_name = [data_name '_rej_ratio'];
saveas(h, figure_name,'png');
saveas(h, figure_name,'pdf');
%save2pdf(figure_name,h,300)
saveas(h, figure_name,'fig');

end

