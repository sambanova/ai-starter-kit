import { LoaderCircle } from "lucide-react";

interface LoadingSpinnerProps {
  message: string;
}

const LoadingSpinner = ({ message }: LoadingSpinnerProps) => {
  return (
    <div className="flex flex-col items-center justify-center space-y-4 my-8">
      <LoaderCircle className="animate-spin h-12 w-12" />
      <div className="text-center text-xl">
        <p className="text-gray-600">{message}</p>
      </div>
    </div>
  );
};

export default LoadingSpinner;
