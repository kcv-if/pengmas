I = imread('ship.jpg');
[row, col] = size(I);
if row > col
    if row > 1000
        m = 1000/row;
    else
        m = 1;
    end
else
    if col > 1000
       m = 1000 / col;
    else
       m = 1;
    end
end
resizeI = imresize(I, m);
grayscl = rgb2gray(resizeI);
prepI = im2double(grayscl);

h = [1;2;1];
v = [1 0 -1];
krnl_x = h*v;
krnl_y = krnl_x.';
Gx = conv2(prepI, krnl_x);
Gy = conv2(prepI, krnl_y);
G_x = Gx + (Gx-imgaussfilt(Gx)) * 0.1;
G_y = Gy + (Gy - imgaussfilt(Gy)) * 0.1;
[Gmag, Gdir] = imgradient(G_x, G_y);
Gmag_thd = Gmag > 0;
sigma = 2;
Amg = imgaussfilt(double(Gmag_thd), sigma);
fuse = imfuse(G_y, G_x, 'blend', 'scaling', 'joint');

Am = double(fuse-(fuse-imgaussfilt(fuse,sigma)));
scl = 20;
[fid,errmsg] = fopen('result.obj', 'w');

med = double(median(fuse(:)));
minim = double(min(fuse(:)));
[rows, columns, numberOfColorChannels] = size(fuse);
for a=1:rows
	for b=1:columns
		if double(Amg(a,b))<0.99
			t = Amg(a,b);
			f=t*med/255*scl;
		elseif Am(a,b)>med
			t = med-(Am(a,b)-med);
			f=t/255*scl;
		else
			t = Am(a,b);
			f=t/255*scl;
        end
	fprintf(fid,'v %f %f %f\r\n',a,b,f);
    end
end
fprintf(fid,'s 1\r\n');
x = rows*columns;
for d = 1:x
	if (d/columns <= rows-1)
		if mod((d+columns),2) ~= 0 && mod(d,columns) ~= 0
			fprintf(fid, 'f %d %d %d\r\n',d,d+1,columns+d);
        end
		if mod((d+columns),2) ~= 0 && mod(d+columns,columns) ~= 1
			fprintf(fid, 'f %d %d %d\r\n',d,d-1,columns+d);
        end
		if mod((d+columns),2) == 0 && mod(d,columns) ~= 1
			fprintf(fid, 'f %d %d %d\r\n',d,columns+d,columns+d-1);
        end
		if mod((d+columns),2) == 0 && mod(d,columns) ~= 0
			fprintf(fid, 'f %d %d %d\r\n',d,columns+d,columns+d+1);
        end
    end
end
fid = fclose(fid);