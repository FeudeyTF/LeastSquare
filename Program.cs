using MathNet.Numerics.LinearAlgebra;


MNK mnk = new(a => Math.Sin(a), a => Math.Sin(2 * a), a => Math.Sin(3 * a));
var parameters = mnk.GetParameters("dots.txt");
var errors = mnk.GetLSE(mnk.SerializePoints("dots.txt"));
for (int i = 0; i < parameters.Count; i++)
    Console.WriteLine(parameters[i] + " +- " + errors[i]);

public delegate T Function<T>(T x);

public class MNK
{
    private List<Function<double>> BaseFunctions;

    public MNK(params Function<double>[] functions) : this(functions.ToList()) { }

    public MNK(List<Function<double>> functions)
    {
        BaseFunctions = functions;
    }

    public List<double> GetParameters(string fileName, string sep = " ") => GetParameters(SerializePoints(fileName, sep));

    public List<double> GetParameters(List<Point<double>> points)
    {
        int N = BaseFunctions.Count;
        int M = points.Count;

        var c = CreateMatrix.Dense<double>(M, N);

        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                c[i, j] = BaseFunctions[j](points[i].X);

        var b = CreateMatrix.Dense<double>(N, N);

        for (int k = 0; k < N; k++)
            for (int j = 0; j < N; j++)
            {
                double value = 0;
                for (int i = 0; i < M; i++)
                    value += c[i, k] * c[i, j];
                b[k, j] = value;
            }

        Vector<double> z = CreateVector.Dense<double>(N);

        for (int k = 0; k < N; k++)
        {
            double value = 0;
            for (int i = 0; i < M; i++)
                value += c[i, k] * points[i].Y;
            z[k] = value;
        }

        return (b.Inverse() * z).ToList();
    }

    public List<double> GetEpsilons(List<Point<double>> points)
    {
        var parameters = GetParameters(points);
        List<double> errors = new();
        for (int i = 0; i < points.Count; i++)
        {
            double value = 0;
            for (int j = 0; j < BaseFunctions.Count; j++)
                value += parameters[j] * BaseFunctions[j](points[i].X);

            value -= points[i].Y;

            errors.Add(value);
        }
        return errors;
    }

    public double GetSigma(List<Point<double>> points)
    {
        var eps = GetEpsilons(points);
        double sum2 = 0;
        foreach (var ep in eps)
            sum2 += ep * ep;
        return sum2 / (points.Count - BaseFunctions.Count);
    }

    public List<double> GetLSE(List<Point<double>> points)
    {
        List<double> result = new();

        int N = BaseFunctions.Count;
        int M = points.Count;

        var c = CreateMatrix.Dense<double>(M, N);

        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                c[i, j] = BaseFunctions[j](points[i].X);

        var b = CreateMatrix.Dense<double>(N, N);

        for (int k = 0; k < N; k++)
            for (int j = 0; j < N; j++)
            {
                double value = 0;
                for (int i = 0; i < M; i++)
                    value += c[i, k] * c[i, j];
                b[k, j] = value;
            }

        b = b.Inverse();

        var sigma0 = Math.Sqrt(GetSigma(points));
        for (int i = 0; i < N; i++)
            result.Add(sigma0 * Math.Sqrt(b[i, i]));

        return result;


    }

    public List<Point<double>> SerializePoints(string fileName, string sep = " ")
    {
        List<Point<double>> points = new();
        var file = File.OpenText(fileName);
        while (!file.EndOfStream)
        {
            string[] dots = file.ReadLine().Split(sep);
            points.Add(new(double.Parse(dots[0]), double.Parse(dots[1])));
        }

        return points;
    }
}


public struct Point<T> where T : struct, IEquatable<T>, IFormattable
{
    public T X;
    public T Y;
    public Point(T x, T y)
    {
        X = x; Y = y;
    }
}